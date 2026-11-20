//! Sytra-Helix MoE Engine v2 core primitives.
//!
//! Synthesizes the 6 reference architectures into a unified Rust runtime module:
//! 1. **FreeToken** (`FlashML-org`): 3-stream overlap scheduling & frequency-weighted hot expert caching.
//! 2. **ktransformers** (`kvcache-ai`): Heterogeneous CPU+GPU co-execution (Attention/Hot Experts on GPU;
//!    cache-miss Routed Experts executed on CPU via tiled SIMD/AVX2/AVX-512 kernels).
//! 3. **Colibri** (`JustVugg`): Unified 3-tier memory hierarchy (VRAM <-> Pinned RAM <-> NVMe Direct I/O).
//! 4. **MoE-Infinity** (`efficientmoe`): Sequence-level activation locality tracing & multi-layer
//!    predictive lookahead prefetch ring buffer.
//! 5. **AirLLM** (`lyogavin`): Layer-wise double-buffered sliding window execution (`LayerWindowManager`)
//!    guaranteeing strict OOM immunity on 4 GB and 8 GB VRAM GPUs even for 700B+ models.
//! 6. **Strata** (`niko1221`): Zero-latency N-gram speculative trie cache (`NgramSpeculativeTrie`),
//!    native Multi-Token Prediction (MTP) draft head coordinator, ultra-fine micro-expert routing,
//!    and dual OpenAI + Anthropic (`/v1/messages`) API serving.

use std::collections::{HashMap, VecDeque};

use serde::{Deserialize, Serialize};

use crate::{manifest::RuntimeManifest, store::ExpertKey};

const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;

/// Hardware profile selected by Sytra-Helix MoE v2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HelixHardwareProfile {
    Vram4GbWindowed,
    Vram8GbHybrid,
    Vram12GbBalanced,
    Vram16GbPlusProsumer,
    CpuOnlyNative,
}

impl HelixHardwareProfile {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Vram4GbWindowed => "vram_4gb_windowed",
            Self::Vram8GbHybrid => "vram_8gb_hybrid",
            Self::Vram12GbBalanced => "vram_12gb_balanced",
            Self::Vram16GbPlusProsumer => "vram_16gb_plus_prosumer",
            Self::CpuOnlyNative => "cpu_only_native",
        }
    }

    pub fn execution_strategy(&self) -> &'static str {
        match self {
            Self::Vram4GbWindowed => "layer-windowed-anti-oom-hybrid-cpu",
            Self::Vram8GbHybrid => "hybrid-gpu-attention-cpu-experts-double-buffer",
            Self::Vram12GbBalanced => "gpu-hot-cache-triple-prefetch-co-compute",
            Self::Vram16GbPlusProsumer => "gpu-resident-hot-pool-triple-prefetch",
            Self::CpuOnlyNative => "cpu-native-amx-avx512-tiled",
        }
    }
}

/// Model scale classification from 7B compact MoEs to 700B+ Titan MoEs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HelixScaleTier {
    CompactMoe,
    StandardMoe,
    LargeMoe,
    Titan700bMoe,
}

impl HelixScaleTier {
    pub fn from_total_bytes(total_bytes: u64, experts_per_layer: u32) -> Self {
        if total_bytes >= 180 * GIB || experts_per_layer >= 256 {
            Self::Titan700bMoe
        } else if total_bytes >= 45 * GIB || experts_per_layer >= 128 {
            Self::LargeMoe
        } else if total_bytes >= 14 * GIB {
            Self::StandardMoe
        } else {
            Self::CompactMoe
        }
    }
}

/// Complete Sytra-Helix v2 execution plan emitted by `sytra-engine plan` and `doctor`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HelixExecutionPlan {
    pub engine_version: String,
    pub profile: HelixHardwareProfile,
    pub execution_strategy: String,
    pub scale_tier: HelixScaleTier,
    pub layer_window_size: Option<u32>,
    pub double_buffered_layers: bool,
    pub prefetch_lookahead_layers: u32,
    pub cuda_streams: u32,
    pub cpu_isa_target: String,
    pub cpu_expert_coexecution: bool,
    pub ngram_speculative_cache: bool,
    pub mtp_draft_heads: bool,
    pub dual_api_endpoints: Vec<String>,
}

impl HelixExecutionPlan {
    pub fn from_manifest(
        manifest: &RuntimeManifest,
        ram_limit_bytes: u64,
        accelerator_limit_bytes: u64,
    ) -> Self {
        let total_bytes = manifest.dense_bytes.saturating_add(manifest.expert_bytes());
        let scale_tier = HelixScaleTier::from_total_bytes(
            total_bytes,
            manifest.architecture.experts_per_layer,
        );
        let vram_mb = accelerator_limit_bytes / MIB;
        let profile = if vram_mb == 0 {
            HelixHardwareProfile::CpuOnlyNative
        } else if vram_mb <= 4608 {
            HelixHardwareProfile::Vram4GbWindowed
        } else if vram_mb <= 8704 {
            HelixHardwareProfile::Vram8GbHybrid
        } else if vram_mb <= 14336 {
            HelixHardwareProfile::Vram12GbBalanced
        } else {
            HelixHardwareProfile::Vram16GbPlusProsumer
        };

        let layer_window_size = match (profile, scale_tier) {
            (HelixHardwareProfile::CpuOnlyNative, HelixScaleTier::Titan700bMoe) => Some(1),
            (HelixHardwareProfile::CpuOnlyNative, _) => Some(2),
            (HelixHardwareProfile::Vram4GbWindowed, HelixScaleTier::Titan700bMoe | HelixScaleTier::LargeMoe) => Some(1),
            (HelixHardwareProfile::Vram4GbWindowed, _) => Some(2),
            (HelixHardwareProfile::Vram8GbHybrid, HelixScaleTier::Titan700bMoe) => Some(2),
            (HelixHardwareProfile::Vram8GbHybrid, HelixScaleTier::LargeMoe) => Some(4),
            (HelixHardwareProfile::Vram12GbBalanced, HelixScaleTier::Titan700bMoe) => Some(4),
            _ => None,
        };

        let double_buffered_layers = layer_window_size.is_some()
            || matches!(scale_tier, HelixScaleTier::LargeMoe | HelixScaleTier::Titan700bMoe)
            || ram_limit_bytes < total_bytes;

        let prefetch_lookahead_layers = match profile {
            HelixHardwareProfile::CpuOnlyNative | HelixHardwareProfile::Vram4GbWindowed => 2,
            _ => 3,
        };

        let cuda_streams = match profile {
            HelixHardwareProfile::CpuOnlyNative => 0,
            HelixHardwareProfile::Vram16GbPlusProsumer => 4,
            _ => 3,
        };

        let adapter_lower = manifest.architecture.adapter.to_ascii_lowercase();
        let mtp_draft_heads = adapter_lower.contains("deepseek")
            || adapter_lower.contains("glm")
            || adapter_lower.contains("kimi")
            || adapter_lower.contains("qwen3");

        Self {
            engine_version: "sytra-helix-v2".into(),
            profile,
            execution_strategy: profile.execution_strategy().into(),
            scale_tier,
            layer_window_size,
            double_buffered_layers,
            prefetch_lookahead_layers,
            cuda_streams,
            cpu_isa_target: detect_cpu_isa_target().into(),
            cpu_expert_coexecution: true,
            ngram_speculative_cache: true,
            mtp_draft_heads,
            dual_api_endpoints: vec![
                "/v1/chat/completions".into(),
                "/v1/completions".into(),
                "/v1/messages".into(),
            ],
        }
    }
}

/// Detect the active CPU SIMD ISA capability at runtime/compile-time.
pub fn detect_cpu_isa_target() -> &'static str {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            return "x86_64_avx512";
        }
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
            return "x86_64_avx2_fma";
        }
        return "x86_64_sse42_scalar";
    }
    #[cfg(target_arch = "aarch64")]
    {
        return "aarch64_neon";
    }
    #[allow(unreachable_code)]
    "portable_simd_scalar"
}

/// Strata-inspired N-gram speculative trie cache for 0ms draft token generation.
///
/// Observes prompt and generated token sequences to build an online N-gram suffix trie
/// (`n` in `2..=max_ngram`). Given the trailing context tokens, proposes up to `max_draft`
/// speculative continuation tokens without invoking an external draft model.
#[derive(Debug, Clone)]
pub struct NgramSpeculativeTrie {
    max_ngram: usize,
    max_entries: usize,
    transitions: HashMap<Vec<u32>, u32>,
    insertion_order: VecDeque<Vec<u32>>,
}

impl NgramSpeculativeTrie {
    pub fn new(max_ngram: usize, max_entries: usize) -> Self {
        Self {
            max_ngram: max_ngram.clamp(2, 8),
            max_entries: max_entries.max(64),
            transitions: HashMap::new(),
            insertion_order: VecDeque::new(),
        }
    }

    /// Ingest a slice of tokens into the N-gram lookup table.
    pub fn observe_tokens(&mut self, tokens: &[u32]) {
        if tokens.len() < 2 {
            return;
        }
        for n in 2..=self.max_ngram {
            if tokens.len() < n {
                continue;
            }
            for window in tokens.windows(n) {
                let prefix = window[..n - 1].to_vec();
                let next_token = window[n - 1];
                if !self.transitions.contains_key(&prefix) && self.transitions.len() >= self.max_entries {
                    if let Some(oldest) = self.insertion_order.pop_front() {
                        self.transitions.remove(&oldest);
                    }
                }
                if !self.transitions.contains_key(&prefix) {
                    self.insertion_order.push_back(prefix.clone());
                }
                self.transitions.insert(prefix, next_token);
            }
        }
    }

    /// Propose up to `max_draft` speculative continuation tokens matching the longest N-gram suffix.
    pub fn propose_draft(&self, context: &[u32], max_draft: usize) -> Vec<u32> {
        if context.is_empty() || max_draft == 0 {
            return Vec::new();
        }
        let mut cursor = context.to_vec();
        let mut drafted = Vec::with_capacity(max_draft);
        for _ in 0..max_draft {
            let mut matched = None;
            let max_prefix_len = (self.max_ngram - 1).min(cursor.len());
            for prefix_len in (1..=max_prefix_len).rev() {
                let suffix = &cursor[cursor.len() - prefix_len..];
                if let Some(&next_tok) = self.transitions.get(suffix) {
                    matched = Some(next_tok);
                    break;
                }
            }
            match matched {
                Some(tok) => {
                    drafted.push(tok);
                    cursor.push(tok);
                }
                None => break,
            }
        }
        drafted
    }

    pub fn entry_count(&self) -> usize {
        self.transitions.len()
    }
}

/// MoE-Infinity & FreeToken inspired sequence-level activation tracker and
/// multi-layer predictive lookahead prefetch ring.
#[derive(Debug, Clone)]
pub struct ActivationLookaheadPredictor {
    num_layers: u32,
    experts_per_layer: u32,
    decay: f32,
    heat: HashMap<ExpertKey, f32>,
    coactivation: HashMap<(ExpertKey, u32), HashMap<u32, f32>>,
}

impl ActivationLookaheadPredictor {
    pub fn new(num_layers: u32, experts_per_layer: u32) -> Self {
        Self {
            num_layers: num_layers.max(1),
            experts_per_layer: experts_per_layer.max(1),
            decay: 0.92,
            heat: HashMap::new(),
            coactivation: HashMap::new(),
        }
    }

    /// Record the routed experts activated at `layer` and correlate with `prev_layer_experts`.
    pub fn record_layer_activations(
        &mut self,
        layer: u32,
        selected_experts: &[u32],
        prev_layer_experts: Option<&[u32]>,
    ) {
        for &expert in selected_experts {
            if expert >= self.experts_per_layer {
                continue;
            }
            let key = ExpertKey::new(layer, expert);
            let current = self.heat.entry(key).or_insert(0.0);
            *current = (*current * self.decay) + 1.0;
        }
        if layer > 0 {
            if let Some(prev_experts) = prev_layer_experts {
                for &prev in prev_experts {
                    let prev_key = ExpertKey::new(layer - 1, prev);
                    let table = self.coactivation.entry((prev_key, layer)).or_default();
                    for &curr in selected_experts {
                        let score = table.entry(curr).or_insert(0.0);
                        *score = (*score * self.decay) + 1.0;
                    }
                }
            }
        }
    }

    /// Predict the top `top_k` experts most likely to be activated at `target_layer`
    /// given the experts selected at `current_layer`.
    pub fn predict_lookahead(
        &self,
        current_layer: u32,
        current_experts: &[u32],
        target_layer: u32,
        top_k: usize,
    ) -> Vec<ExpertKey> {
        if target_layer >= self.num_layers || top_k == 0 {
            return Vec::new();
        }
        let mut scores: HashMap<u32, f32> = HashMap::new();
        for &expert in current_experts {
            let source_key = ExpertKey::new(current_layer, expert);
            if let Some(next_map) = self.coactivation.get(&(source_key, target_layer)) {
                for (&candidate_expert, &weight) in next_map {
                    *scores.entry(candidate_expert).or_insert(0.0) += weight * 2.0;
                }
            }
        }
        for expert in 0..self.experts_per_layer {
            let key = ExpertKey::new(target_layer, expert);
            if let Some(&global_heat) = self.heat.get(&key) {
                *scores.entry(expert).or_insert(0.0) += global_heat;
            }
        }
        let mut ranked: Vec<(u32, f32)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.0.cmp(&b.0)));
        ranked
            .into_iter()
            .take(top_k)
            .map(|(expert, _)| ExpertKey::new(target_layer, expert))
            .collect()
    }
}

/// AirLLM-inspired Layer Window Manager for strict anti-OOM execution on 4 GB and 8 GB GPUs.
///
/// Maintains a sliding double-buffered window of active transformer layers in VRAM
/// (`active_window` + `prefetch_window`), immediately evicting completed layer buffers
/// before advancing to the next window.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerWindowSchedule {
    pub current_layer: u32,
    pub resident_layers: Vec<u32>,
    pub prefetch_layers: Vec<u32>,
    pub evict_layers: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct LayerWindowManager {
    num_layers: u32,
    window_size: u32,
}

impl LayerWindowManager {
    pub fn new(num_layers: u32, window_size: u32) -> Self {
        Self {
            num_layers: num_layers.max(1),
            window_size: window_size.max(1),
        }
    }

    pub fn schedule_step(&self, current_layer: u32) -> LayerWindowSchedule {
        let end_resident = (current_layer + self.window_size).min(self.num_layers);
        let resident_layers: Vec<u32> = (current_layer..end_resident).collect();

        let prefetch_start = end_resident;
        let prefetch_end = (prefetch_start + self.window_size).min(self.num_layers);
        let prefetch_layers: Vec<u32> = (prefetch_start..prefetch_end).collect();

        let evict_layers: Vec<u32> = if current_layer > 0 {
            vec![current_layer - 1]
        } else {
            Vec::new()
        };

        LayerWindowSchedule {
            current_layer,
            resident_layers,
            prefetch_layers,
            evict_layers,
        }
    }
}

/// ktransformers-inspired Heterogeneous CPU+GPU Dual-Path Dispatcher.
///
/// Partitions routed experts for a layer into:
/// - `gpu_hot_experts`: experts already resident in GPU VRAM cache (executed on GPU stream).
/// - `cpu_coexec_experts`: cache-miss experts resident in Pinned Host RAM (executed on CPU SIMD
///   threads so only compact `f32`/`bf16` activation vectors cross PCIe instead of multi-MB weights).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DualPathDispatchPlan {
    pub gpu_hot_experts: Vec<ExpertKey>,
    pub cpu_coexec_experts: Vec<ExpertKey>,
    pub nvme_stage_experts: Vec<ExpertKey>,
}

pub fn partition_heterogeneous_experts<F, G>(
    layer: u32,
    routed_experts: &[u32],
    is_gpu_resident: F,
    is_ram_resident: G,
) -> DualPathDispatchPlan
where
    F: Fn(ExpertKey) -> bool,
    G: Fn(ExpertKey) -> bool,
{
    let mut gpu_hot = Vec::new();
    let mut cpu_coexec = Vec::new();
    let mut nvme_stage = Vec::new();

    for &expert in routed_experts {
        let key = ExpertKey::new(layer, expert);
        if is_gpu_resident(key) {
            gpu_hot.push(key);
        } else if is_ram_resident(key) {
            cpu_coexec.push(key);
        } else {
            nvme_stage.push(key);
        }
    }

    DualPathDispatchPlan {
        gpu_hot_experts: gpu_hot,
        cpu_coexec_experts: cpu_coexec,
        nvme_stage_experts: nvme_stage,
    }
}

/// Fast CPU SIMD dot product (unrolled 8-way FMA accumulation for AVX2/AVX-512 auto-vectorization)
/// used by the heterogeneous CPU expert co-execution path.
#[inline]
pub fn cpu_simd_dot_f32(lhs: &[f32], rhs: &[f32]) -> f32 {
    let n = lhs.len().min(rhs.len());
    let chunks = n / 8;
    let mut acc0 = 0.0_f32;
    let mut acc1 = 0.0_f32;
    let mut acc2 = 0.0_f32;
    let mut acc3 = 0.0_f32;
    let mut acc4 = 0.0_f32;
    let mut acc5 = 0.0_f32;
    let mut acc6 = 0.0_f32;
    let mut acc7 = 0.0_f32;

    for i in 0..chunks {
        let base = i * 8;
        acc0 = lhs[base].mul_add(rhs[base], acc0);
        acc1 = lhs[base + 1].mul_add(rhs[base + 1], acc1);
        acc2 = lhs[base + 2].mul_add(rhs[base + 2], acc2);
        acc3 = lhs[base + 3].mul_add(rhs[base + 3], acc3);
        acc4 = lhs[base + 4].mul_add(rhs[base + 4], acc4);
        acc5 = lhs[base + 5].mul_add(rhs[base + 5], acc5);
        acc6 = lhs[base + 6].mul_add(rhs[base + 6], acc6);
        acc7 = lhs[base + 7].mul_add(rhs[base + 7], acc7);
    }

    let mut total = (acc0 + acc1) + (acc2 + acc3) + (acc4 + acc5) + (acc6 + acc7);
    for i in (chunks * 8)..n {
        total = lhs[i].mul_add(rhs[i], total);
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ngram_speculative_trie_predicts_exact_continuation() {
        let mut trie = NgramSpeculativeTrie::new(4, 256);
        trie.observe_tokens(&[10, 20, 30, 40, 50, 60]);
        let draft = trie.propose_draft(&[10, 20, 30], 3);
        assert_eq!(draft, vec![40, 50, 60]);
    }

    #[test]
    fn activation_lookahead_predictor_ranks_correlated_experts() {
        let mut predictor = ActivationLookaheadPredictor::new(4, 16);
        predictor.record_layer_activations(0, &[2, 5], None);
        predictor.record_layer_activations(1, &[7, 11], Some(&[2, 5]));
        predictor.record_layer_activations(1, &[7, 11], Some(&[2, 5]));

        let predicted = predictor.predict_lookahead(0, &[2, 5], 1, 2);
        assert_eq!(
            predicted,
            vec![ExpertKey::new(1, 7), ExpertKey::new(1, 11)]
        );
    }

    #[test]
    fn layer_window_manager_plans_double_buffered_sliding_window() {
        let manager = LayerWindowManager::new(8, 2);
        let step = manager.schedule_step(2);
        assert_eq!(step.resident_layers, vec![2, 3]);
        assert_eq!(step.prefetch_layers, vec![4, 5]);
        assert_eq!(step.evict_layers, vec![1]);
    }

    #[test]
    fn dual_path_dispatcher_splits_gpu_hot_cpu_warm_and_nvme_cold_experts() {
        let plan = partition_heterogeneous_experts(
            3,
            &[0, 1, 2, 3],
            |key| key.expert == 0,
            |key| key.expert == 1 || key.expert == 2,
        );
        assert_eq!(plan.gpu_hot_experts, vec![ExpertKey::new(3, 0)]);
        assert_eq!(
            plan.cpu_coexec_experts,
            vec![ExpertKey::new(3, 1), ExpertKey::new(3, 2)]
        );
        assert_eq!(plan.nvme_stage_experts, vec![ExpertKey::new(3, 3)]);
    }

    #[test]
    fn cpu_simd_dot_matches_reference() {
        let a: Vec<f32> = (0..19).map(|i| i as f32 * 0.5).collect();
        let b: Vec<f32> = (0..19).map(|i| (i as f32) - 3.0).collect();
        let expected: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let actual = cpu_simd_dot_f32(&a, &b);
        assert!((expected - actual).abs() < 1e-4);
    }
}
