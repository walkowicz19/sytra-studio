//! Native out-of-core MoE primitives for Sytra.
//!
//! This crate owns placement and movement. Architecture adapters still own
//! tensor interpretation, routing semantics, attention, and model kernels.
//! Moving an expert between disk, RAM, and an accelerator must never alter
//! its bytes or the router's selected expert set.

pub mod adapter;
pub mod cache;
pub mod cuda;
pub mod dense;
pub mod draft;
pub mod generation;
pub mod kimi_k27;
pub mod manifest;
pub mod mixtral;
pub mod mixtral_runtime;
pub mod moe;
pub mod oracle;
pub mod performance;
pub mod profile;
pub mod resource;
pub mod runtime;
pub mod scheduler;
pub mod server;
pub mod speculative;
pub mod standard;
pub mod store;
pub mod tokenizer;

pub use adapter::{
    compiled_adapter, compiled_adapters, validate_compiled_contract, validate_forward_contract,
    AdapterDescriptor, AdapterError, ArchitectureKernel, KernelInput, KernelOutput,
};
pub use cache::{
    Accelerator, AcceleratorBuffer, CacheMetrics, NoAccelerator, ResidencyManager, ResidentExpert,
};
pub use cuda::{CudaAccelerator, CudaMemoryMetrics, PackedInt4Bf16View};
pub use dense::{
    bf16_tile_cpu, bf16_tile_matmul_cpu, bf16_transpose_tile_cpu, bf16_transpose_tile_matmul_cpu,
    tiled_bf16_matmul, tiled_bf16_matmul_rows, tiled_bf16_matvec, tiled_bf16_matvec_rows,
    tiled_bf16_transpose_matmul_rows, tiled_bf16_transpose_matvec_rows, tiled_f32_matmul,
    tiled_packed_int4_group32_bf16_matmul, DenseExecutionError, DenseTileMetrics,
    TiledMatmulOutput, TiledMatvecOutput,
};
pub use draft::{DraftError, DraftModel, OpenAiDraftModel};
pub use generation::{
    GenerationConfig, GenerationError, GenerationOutput, GenerationRuntime, KimiGenerator,
    MixtralGenerator, ModelGenerator,
};
pub use kimi_k27::{
    apply_yarn_rope, attention_softmax_scale, decode_i32_le, decode_u32_le, expert_swiglu,
    int4_group32_bf16_matmul_cpu, int4_group_matvec, mla_decode_absorbed_batch_bf16,
    mla_decode_absorbed_bf16, mla_decode_reference, unpack_int4, AbsorbedMlaOutput,
    CompactMlaCacheEntry, CompactMlaKvCache, KimiDecodeState, KimiError, KimiExecutionBackend,
    KimiExpertWeights, KimiK27Config, KimiOneTokenExecutor, KimiSpeculativeOutput, KimiStepMetrics,
    MlaCacheEntry, MlaKvCache, OwnedPackedInt4Matrix, PackedInt4Matrix,
};
pub use manifest::{
    ActivationKind, ArchitectureContract, AttentionContract, AttentionKind, ExpertLocation,
    ExpertStorageManifest, ExpertTensorLayout, ModelFamily, QuantizationContract, RouterContract,
    RouterScoreKind, RouterSemantics, RuntimeManifest, TensorSegment, WeightFormat,
};
pub use mixtral::{MixtralConfig, MixtralError, MixtralExecutor};
pub use mixtral_runtime::{MixtralRuntime, MixtralRuntimeOptions};
pub use moe::{
    apply_activation, apply_standard_rope, bind_gated_expert, decode_float_values,
    gated_expert_reference, rms_norm, route_topk, route_topk_logits, sigmoid,
    standard_attention_decode, standard_attention_decode_window, DenseMatrix, GatedExpertBinding,
    MoERoute, MoeMathError, StandardKvCache, StandardKvEntry,
};
pub use oracle::{
    checkpoint_fingerprint, verify_kimi_oracle, verify_mixtral_oracle, verify_runtime_oracle,
    LogitProbe, OracleCase, OracleCaseReport, OracleError, OracleReport, OracleRuntime,
    OracleSuite, ORACLE_FILE, ORACLE_SCHEMA,
};
pub use performance::{estimate_io_performance, IoPerformanceEstimate, PerformanceError};
pub use profile::{validate_model_config, ProfileError};
pub use resource::{
    plan_memory_envelope, plan_memory_envelope_with_capabilities, KvTier, MemoryEnvelope,
    ResourceError,
};
pub use runtime::{
    KimiOracleOutputs, KimiRuntime, KimiRuntimeMetrics, KimiRuntimeOptions, KimiRuntimePlacement,
    RuntimeError,
};
pub use scheduler::{PreparedLayer, Route, RoutingBatch, SchedulerError, StreamingScheduler};
pub use server::{
    CompletionBackend, InferencePrompt, InferenceRequest, KimiCompletionBackend,
    MixtralCompletionBackend, OpenAiServer, RuntimeCompletionBackend, ServerError,
};
pub use speculative::{verify_greedy, GreedyVerification, SpeculativeController, SpeculativeError};
pub use standard::{
    floating_gated_expert_batch, floating_gated_expert_batch_resident,
    floating_gated_expert_batch_with_cuda, standard_gated_expert_batch_resident, StandardMoeError,
    StandardMoeKvState,
};
pub use store::{
    DenseStoreMetrics, DenseTensorStore, ExpertKey, ExpertPayload, ExpertStore, ResidentTensor,
    StoreMetrics, TensorStoreError, WeightedMirror,
};
pub use tokenizer::{ChatMessage, ModelTokenizer, TokenizerError};
