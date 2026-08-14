//! CUDA Driver API accelerator storage.
//!
//! The driver is loaded dynamically, so Sytra can stage expert bytes in VRAM
//! without Python, Torch, or a CUDA toolkit installation. Architecture
//! adapters still own CUDA compute kernels and tensor interpretation.

use crate::{
    cache::{Accelerator, AcceleratorBuffer},
    store::ExpertKey,
};
use serde::Serialize;

#[derive(Debug, Clone, Copy, Default, Serialize, PartialEq, Eq)]
pub struct CudaMemoryMetrics {
    pub budget_bytes: u64,
    pub live_bytes: u64,
    pub peak_bytes: u64,
    pub budget_denials: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct PackedInt4Bf16View<'a> {
    pub packed: &'a [u32],
    pub scales: &'a [u16],
    pub rows: usize,
    pub cols: usize,
}

#[cfg(target_os = "windows")]
mod platform {
    use std::{
        collections::HashMap,
        ffi::{c_char, c_void},
        mem, ptr,
        sync::{
            atomic::{AtomicU64, Ordering},
            Mutex,
        },
    };

    use super::*;

    type Module = *mut c_void;
    type CuResult = i32;
    type CuDevice = i32;
    type CuContext = *mut c_void;
    type CuDevicePtr = u64;

    type CuInit = unsafe extern "system" fn(u32) -> CuResult;
    type CuDeviceGet = unsafe extern "system" fn(*mut CuDevice, i32) -> CuResult;
    type CuPrimaryRetain = unsafe extern "system" fn(*mut CuContext, CuDevice) -> CuResult;
    type CuPrimaryRelease = unsafe extern "system" fn(CuDevice) -> CuResult;
    type CuCtxSetCurrent = unsafe extern "system" fn(CuContext) -> CuResult;
    type CuMemAlloc = unsafe extern "system" fn(*mut CuDevicePtr, usize) -> CuResult;
    type CuMemcpyHtoD = unsafe extern "system" fn(CuDevicePtr, *const c_void, usize) -> CuResult;
    type CuMemcpyDtoH = unsafe extern "system" fn(*mut c_void, CuDevicePtr, usize) -> CuResult;
    type CuMemFree = unsafe extern "system" fn(CuDevicePtr) -> CuResult;
    type CuModule = *mut c_void;
    type CuFunction = *mut c_void;
    type CuStream = *mut c_void;
    type CuModuleLoadDataEx = unsafe extern "system" fn(
        *mut CuModule,
        *const c_void,
        u32,
        *mut i32,
        *mut *mut c_void,
    ) -> CuResult;
    type CuModuleGetFunction =
        unsafe extern "system" fn(*mut CuFunction, CuModule, *const c_char) -> CuResult;
    type CuLaunchKernel = unsafe extern "system" fn(
        CuFunction,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        CuStream,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CuResult;
    type CuCtxSynchronize = unsafe extern "system" fn() -> CuResult;
    type CuModuleUnload = unsafe extern "system" fn(CuModule) -> CuResult;

    #[link(name = "kernel32")]
    extern "system" {
        fn LoadLibraryA(name: *const c_char) -> Module;
        fn GetProcAddress(module: Module, name: *const c_char) -> *mut c_void;
        fn FreeLibrary(module: Module) -> i32;
    }

    struct CudaApi {
        module: Module,
        init: CuInit,
        device_get: CuDeviceGet,
        primary_retain: CuPrimaryRetain,
        primary_release: CuPrimaryRelease,
        ctx_set_current: CuCtxSetCurrent,
        mem_alloc: CuMemAlloc,
        memcpy_htod: CuMemcpyHtoD,
        memcpy_dtoh: CuMemcpyDtoH,
        mem_free: CuMemFree,
        module_load_data_ex: CuModuleLoadDataEx,
        module_get_function: CuModuleGetFunction,
        launch_kernel: CuLaunchKernel,
        ctx_synchronize: CuCtxSynchronize,
        module_unload: CuModuleUnload,
    }

    // Function pointers and the loaded module are process-global driver state.
    unsafe impl Send for CudaApi {}
    unsafe impl Sync for CudaApi {}

    impl Drop for CudaApi {
        fn drop(&mut self) {
            unsafe {
                FreeLibrary(self.module);
            }
        }
    }

    unsafe fn symbol<T: Copy>(module: Module, name: &'static [u8]) -> Result<T, String> {
        let address = GetProcAddress(module, name.as_ptr().cast());
        if address.is_null() {
            return Err(format!(
                "CUDA driver symbol {} is unavailable",
                String::from_utf8_lossy(&name[..name.len().saturating_sub(1)])
            ));
        }
        Ok(mem::transmute_copy(&address))
    }

    impl CudaApi {
        fn load() -> Result<Self, String> {
            let module = unsafe { LoadLibraryA(c"nvcuda.dll".as_ptr()) };
            if module.is_null() {
                return Err("nvcuda.dll is unavailable; install an NVIDIA driver".into());
            }
            let result = unsafe {
                Ok(Self {
                    module,
                    init: symbol(module, b"cuInit\0")?,
                    device_get: symbol(module, b"cuDeviceGet\0")?,
                    primary_retain: symbol(module, b"cuDevicePrimaryCtxRetain\0")?,
                    primary_release: symbol(module, b"cuDevicePrimaryCtxRelease_v2\0")
                        .or_else(|_| symbol(module, b"cuDevicePrimaryCtxRelease\0"))?,
                    ctx_set_current: symbol(module, b"cuCtxSetCurrent\0")?,
                    mem_alloc: symbol(module, b"cuMemAlloc_v2\0")
                        .or_else(|_| symbol(module, b"cuMemAlloc\0"))?,
                    memcpy_htod: symbol(module, b"cuMemcpyHtoD_v2\0")
                        .or_else(|_| symbol(module, b"cuMemcpyHtoD\0"))?,
                    memcpy_dtoh: symbol(module, b"cuMemcpyDtoH_v2\0")
                        .or_else(|_| symbol(module, b"cuMemcpyDtoH\0"))?,
                    mem_free: symbol(module, b"cuMemFree_v2\0")
                        .or_else(|_| symbol(module, b"cuMemFree\0"))?,
                    module_load_data_ex: symbol(module, b"cuModuleLoadDataEx\0")?,
                    module_get_function: symbol(module, b"cuModuleGetFunction\0")?,
                    launch_kernel: symbol(module, b"cuLaunchKernel\0")?,
                    ctx_synchronize: symbol(module, b"cuCtxSynchronize\0")?,
                    module_unload: symbol(module, b"cuModuleUnload\0")?,
                })
            };
            if result.is_err() {
                unsafe {
                    FreeLibrary(module);
                }
            }
            result
        }
    }

    pub struct CudaAccelerator {
        api: CudaApi,
        device: CuDevice,
        context: CuContext,
        allocations: Mutex<HashMap<CuDevicePtr, u64>>,
        kernels: Mutex<HashMap<&'static str, (CuModule, CuFunction)>>,
        memory_budget: u64,
        live_bytes: AtomicU64,
        peak_bytes: AtomicU64,
        budget_denials: AtomicU64,
    }

    unsafe impl Send for CudaAccelerator {}
    unsafe impl Sync for CudaAccelerator {}

    impl std::fmt::Debug for CudaAccelerator {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter
                .debug_struct("CudaAccelerator")
                .field("device", &self.device)
                .field("memory", &self.memory_metrics())
                .field(
                    "allocations",
                    &self
                        .allocations
                        .lock()
                        .map(|value| value.len())
                        .unwrap_or(0),
                )
                .finish()
        }
    }

    impl CudaAccelerator {
        pub fn new(ordinal: i32) -> Result<Self, String> {
            Self::new_with_budget(ordinal, u64::MAX)
        }

        pub fn new_with_budget(ordinal: i32, memory_budget: u64) -> Result<Self, String> {
            if memory_budget == 0 {
                return Err("CUDA allocation budget must be positive".into());
            }
            let api = CudaApi::load()?;
            let mut device = 0;
            let mut context = ptr::null_mut();
            unsafe {
                check((api.init)(0), "cuInit")?;
                check((api.device_get)(&mut device, ordinal), "cuDeviceGet")?;
                check(
                    (api.primary_retain)(&mut context, device),
                    "cuDevicePrimaryCtxRetain",
                )?;
                check((api.ctx_set_current)(context), "cuCtxSetCurrent")?;
            }
            Ok(Self {
                api,
                device,
                context,
                allocations: Mutex::new(HashMap::new()),
                kernels: Mutex::new(HashMap::new()),
                memory_budget,
                live_bytes: AtomicU64::new(0),
                peak_bytes: AtomicU64::new(0),
                budget_denials: AtomicU64::new(0),
            })
        }

        pub fn memory_metrics(&self) -> CudaMemoryMetrics {
            CudaMemoryMetrics {
                budget_bytes: self.memory_budget,
                live_bytes: self.live_bytes.load(Ordering::Acquire),
                peak_bytes: self.peak_bytes.load(Ordering::Acquire),
                budget_denials: self.budget_denials.load(Ordering::Acquire),
            }
        }

        /// Reserve device memory without creating a same-sized host zero buffer.
        /// Callers must initialize every byte they read through `write_buffer`.
        pub fn allocate(&self, bytes: usize) -> Result<AcceleratorBuffer, String> {
            if bytes == 0 {
                return Err("cannot allocate an empty CUDA buffer".into());
            }
            let size =
                u64::try_from(bytes).map_err(|_| "CUDA allocation size exceeds u64".to_string())?;
            if !reserve_global_bytes(&self.live_bytes, &self.peak_bytes, size, self.memory_budget) {
                self.budget_denials.fetch_add(1, Ordering::AcqRel);
                return Err(format!(
                    "CUDA allocation needs {size} bytes but the hard device limit is {}",
                    self.memory_budget
                ));
            }
            let mut pointer = 0;
            let allocation = (|| unsafe {
                check((self.api.ctx_set_current)(self.context), "cuCtxSetCurrent")?;
                check((self.api.mem_alloc)(&mut pointer, bytes), "cuMemAlloc")?;
                Ok::<(), String>(())
            })();
            if let Err(error) = allocation {
                self.live_bytes.fetch_sub(size, Ordering::AcqRel);
                return Err(error);
            }
            let mut allocations = match self.allocations.lock() {
                Ok(allocations) => allocations,
                Err(_) => {
                    unsafe {
                        let _ = (self.api.mem_free)(pointer);
                    }
                    self.live_bytes.fetch_sub(size, Ordering::AcqRel);
                    return Err("CUDA allocation registry is poisoned".into());
                }
            };
            allocations.insert(pointer, size);
            Ok(AcceleratorBuffer { id: pointer, bytes })
        }

        /// Update one validated byte range inside a live CUDA allocation.
        pub fn write_buffer(
            &self,
            buffer: &AcceleratorBuffer,
            offset: usize,
            bytes: &[u8],
        ) -> Result<(), String> {
            let end = offset
                .checked_add(bytes.len())
                .ok_or_else(|| "CUDA write range overflow".to_string())?;
            let registered = self
                .allocations
                .lock()
                .map_err(|_| "CUDA allocation registry is poisoned".to_string())?
                .get(&buffer.id)
                .copied();
            if registered != Some(buffer.bytes as u64) || end > buffer.bytes {
                return Err("CUDA write exceeds a live allocation".into());
            }
            unsafe {
                check((self.api.ctx_set_current)(self.context), "cuCtxSetCurrent")?;
                check(
                    (self.api.memcpy_htod)(
                        buffer.id + offset as u64,
                        bytes.as_ptr().cast::<c_void>(),
                        bytes.len(),
                    ),
                    "cuMemcpyHtoD",
                )
            }
        }

        fn kernel_function(
            &self,
            key: &'static str,
            ptx_source: &'static [u8],
            symbol_name: *const c_char,
        ) -> Result<CuFunction, String> {
            let mut kernels = self
                .kernels
                .lock()
                .map_err(|_| "CUDA kernel registry is poisoned".to_string())?;
            if let Some((_, function)) = kernels.get(key) {
                return Ok(*function);
            }
            let mut module = ptr::null_mut();
            let mut function = ptr::null_mut();
            let mut ptx = ptx_source.to_vec();
            ptx.push(0);
            unsafe {
                check((self.api.ctx_set_current)(self.context), "cuCtxSetCurrent")?;
                check(
                    (self.api.module_load_data_ex)(
                        &mut module,
                        ptx.as_ptr().cast(),
                        0,
                        ptr::null_mut(),
                        ptr::null_mut(),
                    ),
                    "cuModuleLoadDataEx(cached PTX)",
                )?;
                if let Err(error) = check(
                    (self.api.module_get_function)(&mut function, module, symbol_name),
                    "cuModuleGetFunction",
                ) {
                    let _ = (self.api.module_unload)(module);
                    return Err(error);
                }
            }
            kernels.insert(key, (module, function));
            Ok(function)
        }

        /// Correctness-first CUDA kernel for compressed-tensors symmetric
        /// packed INT4 group-32 matrix-vector products.
        ///
        /// This establishes the exact GPU decoding contract. Optimized
        /// tensor-core kernels can replace it only after matching its output
        /// and the architecture reference oracle.
        pub fn int4_group32_matvec(
            &self,
            packed: &[u32],
            scales: &[f32],
            rows: usize,
            cols: usize,
            input: &[f32],
        ) -> Result<Vec<f32>, String> {
            if rows == 0
                || cols == 0
                || cols % 32 != 0
                || input.len() != cols
                || packed.len() != rows * cols.div_ceil(8)
                || scales.len() != rows * (cols / 32)
            {
                return Err("invalid CUDA INT4 group-32 matvec dimensions".into());
            }
            let packed_bytes = as_bytes(packed);
            let scale_bytes = as_bytes(scales);
            let input_bytes = as_bytes(input);
            let output_zero = vec![0.0_f32; rows];
            let packed_buffer = self.upload(ExpertKey::new(0, 0), packed_bytes)?;
            let scale_buffer = match self.upload(ExpertKey::new(0, 1), scale_bytes) {
                Ok(buffer) => buffer,
                Err(error) => {
                    self.release(&packed_buffer);
                    return Err(error);
                }
            };
            let input_buffer = match self.upload(ExpertKey::new(0, 2), input_bytes) {
                Ok(buffer) => buffer,
                Err(error) => {
                    self.release(&scale_buffer);
                    self.release(&packed_buffer);
                    return Err(error);
                }
            };
            let output_buffer = match self.upload(ExpertKey::new(0, 3), as_bytes(&output_zero)) {
                Ok(buffer) => buffer,
                Err(error) => {
                    self.release(&input_buffer);
                    self.release(&scale_buffer);
                    self.release(&packed_buffer);
                    return Err(error);
                }
            };

            let result = self.launch_int4_group32(
                packed_buffer.id,
                scale_buffer.id,
                input_buffer.id,
                output_buffer.id,
                rows,
                cols,
            );
            self.release(&output_buffer);
            self.release(&input_buffer);
            self.release(&scale_buffer);
            self.release(&packed_buffer);
            result
        }

        /// Same reference kernel with the checkpoint's native BF16 scale
        /// tensor format. No scale expansion is staged in RAM or VRAM.
        pub fn int4_group32_bf16_matvec(
            &self,
            packed: &[u32],
            scales: &[u16],
            rows: usize,
            cols: usize,
            input: &[f32],
        ) -> Result<Vec<f32>, String> {
            if rows == 0
                || cols == 0
                || cols % 32 != 0
                || input.len() != cols
                || packed.len() != rows * cols.div_ceil(8)
                || scales.len() != rows * (cols / 32)
            {
                return Err("invalid CUDA INT4/BF16 group-32 matvec dimensions".into());
            }
            let output_zero = vec![0.0_f32; rows];
            let payloads = [
                as_bytes(packed),
                as_bytes(scales),
                as_bytes(input),
                as_bytes(&output_zero),
            ];
            let mut buffers = Vec::with_capacity(payloads.len());
            for (index, payload) in payloads.into_iter().enumerate() {
                match self.upload(ExpertKey::new(0, index as u32), payload) {
                    Ok(buffer) => buffers.push(buffer),
                    Err(error) => {
                        for buffer in buffers.iter().rev() {
                            self.release(buffer);
                        }
                        return Err(error);
                    }
                }
            }
            let result = self.launch_int4_group32_bf16(
                buffers[0].id,
                buffers[1].id,
                buffers[2].id,
                buffers[3].id,
                rows,
                cols,
            );
            for buffer in buffers.iter().rev() {
                self.release(buffer);
            }
            result
        }

        /// Batched packed INT4/BF16 projection with position-major FP32
        /// activations. Packed weights and scales are uploaded once for the
        /// complete batch instead of once per speculative position.
        pub fn int4_group32_bf16_bytes_matmul(
            &self,
            packed: &[u8],
            scales: &[u8],
            rows: usize,
            cols: usize,
            positions: usize,
            input: &[f32],
        ) -> Result<Vec<f32>, String> {
            let packed_bytes = rows
                .checked_mul(cols.div_ceil(8))
                .and_then(|words| words.checked_mul(size_of::<u32>()))
                .ok_or_else(|| "packed INT4 batch size overflow".to_string())?;
            let scale_bytes = rows
                .checked_mul(cols / 32)
                .and_then(|values| values.checked_mul(size_of::<u16>()))
                .ok_or_else(|| "BF16 scale batch size overflow".to_string())?;
            let input_elements = positions
                .checked_mul(cols)
                .ok_or_else(|| "INT4 batch input size overflow".to_string())?;
            let output_elements = positions
                .checked_mul(rows)
                .ok_or_else(|| "INT4 batch output size overflow".to_string())?;
            if rows == 0
                || cols == 0
                || positions == 0
                || cols % 32 != 0
                || packed.len() != packed_bytes
                || scales.len() != scale_bytes
                || input.len() != input_elements
            {
                return Err("invalid CUDA INT4/BF16 batched matmul dimensions".into());
            }
            let output_zero = vec![0.0_f32; output_elements];
            let payloads = [packed, scales, as_bytes(input), as_bytes(&output_zero)];
            let mut buffers = Vec::with_capacity(payloads.len());
            for (index, payload) in payloads.into_iter().enumerate() {
                match self.upload(ExpertKey::new(0, index as u32), payload) {
                    Ok(buffer) => buffers.push(buffer),
                    Err(error) => {
                        for buffer in buffers.iter().rev() {
                            self.release(buffer);
                        }
                        return Err(error);
                    }
                }
            }
            let result = self.launch_int4_group32_bf16_matmul(
                buffers[0].id,
                buffers[1].id,
                buffers[2].id,
                buffers[3].id,
                rows,
                cols,
                positions,
            );
            for buffer in buffers.iter().rev() {
                self.release(buffer);
            }
            result
        }

        /// Execute a packed INT4/BF16 projection directly from a byte range
        /// inside an expert allocation already resident in VRAM. Only the
        /// activation and output buffers are transient; weights are not
        /// duplicated or copied back through the host.
        pub fn resident_int4_group32_bf16_matvec(
            &self,
            resident: &AcceleratorBuffer,
            packed_offset: usize,
            scale_offset: usize,
            rows: usize,
            cols: usize,
            input: &[f32],
        ) -> Result<Vec<f32>, String> {
            if rows == 0 || cols == 0 || cols % 32 != 0 || input.len() != cols {
                return Err("invalid resident CUDA INT4/BF16 matvec dimensions".into());
            }
            let packed_bytes = rows
                .checked_mul(cols.div_ceil(8))
                .and_then(|words| words.checked_mul(size_of::<u32>()))
                .ok_or_else(|| "resident packed tensor size overflow".to_string())?;
            let scale_bytes = rows
                .checked_mul(cols / 32)
                .and_then(|values| values.checked_mul(size_of::<u16>()))
                .ok_or_else(|| "resident scale tensor size overflow".to_string())?;
            if packed_offset
                .checked_add(packed_bytes)
                .filter(|end| *end <= resident.bytes)
                .is_none()
                || scale_offset
                    .checked_add(scale_bytes)
                    .filter(|end| *end <= resident.bytes)
                    .is_none()
            {
                return Err("resident expert tensor range exceeds its CUDA allocation".into());
            }
            let output_zero = vec![0.0_f32; rows];
            let input_buffer = self.upload(ExpertKey::new(0, 0), as_bytes(input))?;
            let output_buffer = match self.upload(ExpertKey::new(0, 1), as_bytes(&output_zero)) {
                Ok(buffer) => buffer,
                Err(error) => {
                    self.release(&input_buffer);
                    return Err(error);
                }
            };
            let result = self.launch_int4_group32_bf16(
                resident.id + packed_offset as u64,
                resident.id + scale_offset as u64,
                input_buffer.id,
                output_buffer.id,
                rows,
                cols,
            );
            self.release(&output_buffer);
            self.release(&input_buffer);
            result
        }

        /// Batched counterpart to the resident matvec. The expert allocation
        /// remains in place; only gathered activations and outputs are
        /// transient.
        pub fn resident_int4_group32_bf16_matmul(
            &self,
            resident: &AcceleratorBuffer,
            packed_offset: usize,
            scale_offset: usize,
            rows: usize,
            cols: usize,
            positions: usize,
            input: &[f32],
        ) -> Result<Vec<f32>, String> {
            let packed_bytes = rows
                .checked_mul(cols.div_ceil(8))
                .and_then(|words| words.checked_mul(size_of::<u32>()))
                .ok_or_else(|| "resident packed tensor size overflow".to_string())?;
            let scale_bytes = rows
                .checked_mul(cols / 32)
                .and_then(|values| values.checked_mul(size_of::<u16>()))
                .ok_or_else(|| "resident scale tensor size overflow".to_string())?;
            let input_elements = positions
                .checked_mul(cols)
                .ok_or_else(|| "resident INT4 batch input overflow".to_string())?;
            let output_elements = positions
                .checked_mul(rows)
                .ok_or_else(|| "resident INT4 batch output overflow".to_string())?;
            if rows == 0
                || cols == 0
                || positions == 0
                || cols % 32 != 0
                || input.len() != input_elements
                || packed_offset
                    .checked_add(packed_bytes)
                    .filter(|end| *end <= resident.bytes)
                    .is_none()
                || scale_offset
                    .checked_add(scale_bytes)
                    .filter(|end| *end <= resident.bytes)
                    .is_none()
            {
                return Err("invalid resident CUDA INT4/BF16 batched matmul dimensions".into());
            }
            let output_zero = vec![0.0_f32; output_elements];
            let input_buffer = self.upload(ExpertKey::new(0, 0), as_bytes(input))?;
            let output_buffer = match self.upload(ExpertKey::new(0, 1), as_bytes(&output_zero)) {
                Ok(buffer) => buffer,
                Err(error) => {
                    self.release(&input_buffer);
                    return Err(error);
                }
            };
            let result = self.launch_int4_group32_bf16_matmul(
                resident.id + packed_offset as u64,
                resident.id + scale_offset as u64,
                input_buffer.id,
                output_buffer.id,
                rows,
                cols,
                positions,
            );
            self.release(&output_buffer);
            self.release(&input_buffer);
            result
        }

        /// Portable BF16 weight / FP32 activation matrix-vector product.
        /// `weights_bf16` stays byte-packed so a dense tile is never expanded
        /// to FP32 in host memory before upload.
        pub fn bf16_matvec_bytes(
            &self,
            weights_bf16: &[u8],
            rows: usize,
            cols: usize,
            input: &[f32],
        ) -> Result<Vec<f32>, String> {
            let expected = rows
                .checked_mul(cols)
                .and_then(|elements| elements.checked_mul(2))
                .ok_or_else(|| "BF16 matrix size overflow".to_string())?;
            if rows == 0 || cols == 0 || input.len() != cols || weights_bf16.len() != expected {
                return Err("invalid CUDA BF16 matvec dimensions".into());
            }
            let output_zero = vec![0.0_f32; rows];
            let payloads = [weights_bf16, as_bytes(input), as_bytes(&output_zero)];
            let mut buffers = Vec::with_capacity(payloads.len());
            for (index, payload) in payloads.into_iter().enumerate() {
                match self.upload(ExpertKey::new(0, index as u32), payload) {
                    Ok(buffer) => buffers.push(buffer),
                    Err(error) => {
                        for buffer in buffers.iter().rev() {
                            self.release(buffer);
                        }
                        return Err(error);
                    }
                }
            }
            let result =
                self.launch_bf16_matvec(buffers[0].id, buffers[1].id, buffers[2].id, rows, cols);
            for buffer in buffers.iter().rev() {
                self.release(buffer);
            }
            result
        }

        /// Position-major batched BF16 matrix multiplication used by target
        /// verification. The weight tile is uploaded once for all positions.
        pub fn bf16_matmul_bytes(
            &self,
            weights_bf16: &[u8],
            rows: usize,
            cols: usize,
            positions: usize,
            input: &[f32],
        ) -> Result<Vec<f32>, String> {
            let expected = rows
                .checked_mul(cols)
                .and_then(|elements| elements.checked_mul(2))
                .ok_or_else(|| "BF16 matrix size overflow".to_string())?;
            let input_elements = positions
                .checked_mul(cols)
                .ok_or_else(|| "BF16 batch input size overflow".to_string())?;
            let output_elements = positions
                .checked_mul(rows)
                .ok_or_else(|| "BF16 batch output size overflow".to_string())?;
            if rows == 0
                || cols == 0
                || positions == 0
                || input.len() != input_elements
                || weights_bf16.len() != expected
            {
                return Err("invalid CUDA BF16 batched matmul dimensions".into());
            }
            let output_zero = vec![0.0_f32; output_elements];
            let payloads = [weights_bf16, as_bytes(input), as_bytes(&output_zero)];
            let mut buffers = Vec::with_capacity(payloads.len());
            for (index, payload) in payloads.into_iter().enumerate() {
                match self.upload(ExpertKey::new(0, index as u32), payload) {
                    Ok(buffer) => buffers.push(buffer),
                    Err(error) => {
                        for buffer in buffers.iter().rev() {
                            self.release(buffer);
                        }
                        return Err(error);
                    }
                }
            }
            let result = self.launch_bf16_matmul(
                buffers[0].id,
                buffers[1].id,
                buffers[2].id,
                rows,
                cols,
                positions,
            );
            for buffer in buffers.iter().rev() {
                self.release(buffer);
            }
            result
        }

        /// Execute a BF16 matrix batch directly from a tensor range inside an
        /// expert allocation already resident in VRAM. Only FP32 activation
        /// and output buffers are transient.
        pub fn resident_bf16_matmul(
            &self,
            resident: &AcceleratorBuffer,
            weight_offset: usize,
            rows: usize,
            cols: usize,
            positions: usize,
            input: &[f32],
        ) -> Result<Vec<f32>, String> {
            let weight_bytes = rows
                .checked_mul(cols)
                .and_then(|elements| elements.checked_mul(size_of::<u16>()))
                .ok_or_else(|| "resident BF16 matrix size overflow".to_string())?;
            let input_elements = positions
                .checked_mul(cols)
                .ok_or_else(|| "resident BF16 batch input overflow".to_string())?;
            let output_elements = positions
                .checked_mul(rows)
                .ok_or_else(|| "resident BF16 batch output overflow".to_string())?;
            if rows == 0
                || cols == 0
                || positions == 0
                || input.len() != input_elements
                || weight_offset
                    .checked_add(weight_bytes)
                    .filter(|end| *end <= resident.bytes)
                    .is_none()
            {
                return Err("invalid resident CUDA BF16 batched matmul dimensions".into());
            }
            let output_zero = vec![0.0_f32; output_elements];
            let input_buffer = self.upload(ExpertKey::new(0, 0), as_bytes(input))?;
            let output_buffer = match self.upload(ExpertKey::new(0, 1), as_bytes(&output_zero)) {
                Ok(buffer) => buffer,
                Err(error) => {
                    self.release(&input_buffer);
                    return Err(error);
                }
            };
            let result = self.launch_bf16_matmul(
                resident.id + weight_offset as u64,
                input_buffer.id,
                output_buffer.id,
                rows,
                cols,
                positions,
            );
            self.release(&output_buffer);
            self.release(&input_buffer);
            result
        }

        /// Portable transpose BF16 weight / FP32 activation product used by
        /// the absorbed MLA query path. Output length is `cols`.
        pub fn bf16_transpose_matvec_bytes(
            &self,
            weights_bf16: &[u8],
            rows: usize,
            cols: usize,
            input: &[f32],
        ) -> Result<Vec<f32>, String> {
            let expected = rows
                .checked_mul(cols)
                .and_then(|elements| elements.checked_mul(2))
                .ok_or_else(|| "BF16 transpose matrix size overflow".to_string())?;
            if rows == 0 || cols == 0 || input.len() != rows || weights_bf16.len() != expected {
                return Err("invalid CUDA BF16 transpose matvec dimensions".into());
            }
            let output_zero = vec![0.0_f32; cols];
            let payloads = [weights_bf16, as_bytes(input), as_bytes(&output_zero)];
            let mut buffers = Vec::with_capacity(payloads.len());
            for (index, payload) in payloads.into_iter().enumerate() {
                match self.upload(ExpertKey::new(0, index as u32), payload) {
                    Ok(buffer) => buffers.push(buffer),
                    Err(error) => {
                        for buffer in buffers.iter().rev() {
                            self.release(buffer);
                        }
                        return Err(error);
                    }
                }
            }
            let result = self.launch_bf16_transpose_matvec(
                buffers[0].id,
                buffers[1].id,
                buffers[2].id,
                rows,
                cols,
            );
            for buffer in buffers.iter().rev() {
                self.release(buffer);
            }
            result
        }

        pub fn bf16_transpose_matmul_bytes(
            &self,
            weights_bf16: &[u8],
            rows: usize,
            cols: usize,
            positions: usize,
            input: &[f32],
        ) -> Result<Vec<f32>, String> {
            let expected = rows
                .checked_mul(cols)
                .and_then(|elements| elements.checked_mul(2))
                .ok_or_else(|| "BF16 transpose matrix size overflow".to_string())?;
            let input_elements = positions
                .checked_mul(rows)
                .ok_or_else(|| "BF16 transpose batch input overflow".to_string())?;
            let output_elements = positions
                .checked_mul(cols)
                .ok_or_else(|| "BF16 transpose batch output overflow".to_string())?;
            if rows == 0
                || cols == 0
                || positions == 0
                || input.len() != input_elements
                || weights_bf16.len() != expected
            {
                return Err("invalid CUDA BF16 transpose batched matmul dimensions".into());
            }
            let output_zero = vec![0.0_f32; output_elements];
            let payloads = [weights_bf16, as_bytes(input), as_bytes(&output_zero)];
            let mut buffers = Vec::with_capacity(payloads.len());
            for (index, payload) in payloads.into_iter().enumerate() {
                match self.upload(ExpertKey::new(0, index as u32), payload) {
                    Ok(buffer) => buffers.push(buffer),
                    Err(error) => {
                        for buffer in buffers.iter().rev() {
                            self.release(buffer);
                        }
                        return Err(error);
                    }
                }
            }
            let result = self.launch_bf16_transpose_matmul(
                buffers[0].id,
                buffers[1].id,
                buffers[2].id,
                rows,
                cols,
                positions,
            );
            for buffer in buffers.iter().rev() {
                self.release(buffer);
            }
            result
        }

        /// Correctness-first GPU execution of one Kimi routed expert.
        ///
        /// Gate/up/down projections execute through the native packed
        /// INT4/BF16 CUDA kernel. The SiLU product currently crosses the host;
        /// the fused persistent kernel must match this path before replacing
        /// it.
        /// Decode one MHA/GQA query against a persistent BF16 KV allocation.
        /// Only the query and result cross PCIe; cached keys and values remain
        /// compact and device resident for the request lifetime.
        #[allow(clippy::too_many_arguments)]
        pub fn standard_attention_bf16(
            &self,
            keys: &AcceleratorBuffer,
            values: &AcceleratorBuffer,
            capacity: usize,
            cache_len: usize,
            query: &[f32],
            heads: usize,
            kv_heads: usize,
            head_dim: usize,
            value_dim: usize,
            softmax_scale: f32,
            window: Option<usize>,
        ) -> Result<Vec<f32>, String> {
            let key_bytes = capacity
                .checked_mul(kv_heads)
                .and_then(|count| count.checked_mul(head_dim))
                .and_then(|count| count.checked_mul(size_of::<u16>()))
                .ok_or_else(|| "CUDA KV key size overflow".to_string())?;
            let value_bytes = capacity
                .checked_mul(kv_heads)
                .and_then(|count| count.checked_mul(value_dim))
                .and_then(|count| count.checked_mul(size_of::<u16>()))
                .ok_or_else(|| "CUDA KV value size overflow".to_string())?;
            if capacity == 0
                || cache_len == 0
                || cache_len > capacity
                || heads == 0
                || kv_heads == 0
                || !heads.is_multiple_of(kv_heads)
                || query.len() != heads.saturating_mul(head_dim)
                || keys.bytes != key_bytes
                || values.bytes != value_bytes
                || !softmax_scale.is_finite()
                || softmax_scale <= 0.0
            {
                return Err("invalid CUDA BF16 standard-attention dimensions".into());
            }
            let output = vec![0.0_f32; heads * value_dim];
            let query_buffer = self.upload(ExpertKey::new(0, 0), as_bytes(query))?;
            let output_buffer = match self.upload(ExpertKey::new(0, 1), as_bytes(&output)) {
                Ok(buffer) => buffer,
                Err(error) => {
                    self.release(&query_buffer);
                    return Err(error);
                }
            };
            let result = self.launch_standard_attention_bf16(
                keys.id,
                values.id,
                query_buffer.id,
                output_buffer.id,
                cache_len,
                heads,
                kv_heads,
                head_dim,
                value_dim,
                softmax_scale,
                window.unwrap_or(0),
            );
            self.release(&output_buffer);
            self.release(&query_buffer);
            result
        }

        pub fn expert_swiglu_bf16(
            &self,
            hidden: &[f32],
            gate: PackedInt4Bf16View<'_>,
            up: PackedInt4Bf16View<'_>,
            down: PackedInt4Bf16View<'_>,
        ) -> Result<Vec<f32>, String> {
            if gate.cols != hidden.len()
                || up.cols != hidden.len()
                || gate.rows != up.rows
                || down.cols != gate.rows
            {
                return Err("incompatible CUDA expert projection shapes".into());
            }
            let mut gate_output = self.int4_group32_bf16_matvec(
                gate.packed,
                gate.scales,
                gate.rows,
                gate.cols,
                hidden,
            )?;
            let up_output =
                self.int4_group32_bf16_matvec(up.packed, up.scales, up.rows, up.cols, hidden)?;
            for (gate, up) in gate_output.iter_mut().zip(up_output) {
                let silu = *gate / (1.0 + (-*gate).exp());
                *gate = silu * up;
            }
            self.int4_group32_bf16_matvec(
                down.packed,
                down.scales,
                down.rows,
                down.cols,
                &gate_output,
            )
        }

        fn launch_int4_group32(
            &self,
            packed: CuDevicePtr,
            scales: CuDevicePtr,
            input: CuDevicePtr,
            output: CuDevicePtr,
            rows: usize,
            cols: usize,
        ) -> Result<Vec<f32>, String> {
            let rows_u32 =
                u32::try_from(rows).map_err(|_| "row count exceeds CUDA u32".to_string())?;
            let cols_u32 =
                u32::try_from(cols).map_err(|_| "column count exceeds CUDA u32".to_string())?;
            let function = self.kernel_function(
                "int4_group32_f32",
                INT4_GROUP32_PTX,
                c"sytra_int4_group32_matvec".as_ptr(),
            )?;
            unsafe {
                check((self.api.ctx_set_current)(self.context), "cuCtxSetCurrent")?;
                let operation = (|| {
                    let mut packed_arg = packed;
                    let mut scales_arg = scales;
                    let mut input_arg = input;
                    let mut output_arg = output;
                    let mut rows_arg = rows_u32;
                    let mut cols_arg = cols_u32;
                    let mut arguments = [
                        (&mut packed_arg as *mut CuDevicePtr).cast::<c_void>(),
                        (&mut scales_arg as *mut CuDevicePtr).cast::<c_void>(),
                        (&mut input_arg as *mut CuDevicePtr).cast::<c_void>(),
                        (&mut output_arg as *mut CuDevicePtr).cast::<c_void>(),
                        (&mut rows_arg as *mut u32).cast::<c_void>(),
                        (&mut cols_arg as *mut u32).cast::<c_void>(),
                    ];
                    let threads = 128_u32;
                    let blocks = rows_u32.div_ceil(threads);
                    check(
                        (self.api.launch_kernel)(
                            function,
                            blocks,
                            1,
                            1,
                            threads,
                            1,
                            1,
                            0,
                            ptr::null_mut(),
                            arguments.as_mut_ptr(),
                            ptr::null_mut(),
                        ),
                        "cuLaunchKernel(INT4 group-32)",
                    )?;
                    check((self.api.ctx_synchronize)(), "cuCtxSynchronize")?;
                    let mut result = vec![0.0_f32; rows];
                    check(
                        (self.api.memcpy_dtoh)(
                            result.as_mut_ptr().cast(),
                            output,
                            rows * size_of::<f32>(),
                        ),
                        "cuMemcpyDtoH",
                    )?;
                    Ok(result)
                })();
                operation
            }
        }

        fn launch_int4_group32_bf16(
            &self,
            packed: CuDevicePtr,
            scales: CuDevicePtr,
            input: CuDevicePtr,
            output: CuDevicePtr,
            rows: usize,
            cols: usize,
        ) -> Result<Vec<f32>, String> {
            let rows_u32 =
                u32::try_from(rows).map_err(|_| "row count exceeds CUDA u32".to_string())?;
            let cols_u32 =
                u32::try_from(cols).map_err(|_| "column count exceeds CUDA u32".to_string())?;
            let function = self.kernel_function(
                "int4_group32_bf16",
                INT4_GROUP32_BF16_PTX,
                c"sytra_int4_group32_bf16_matvec".as_ptr(),
            )?;
            unsafe {
                check((self.api.ctx_set_current)(self.context), "cuCtxSetCurrent")?;
                let operation = (|| {
                    let mut packed_arg = packed;
                    let mut scales_arg = scales;
                    let mut input_arg = input;
                    let mut output_arg = output;
                    let mut rows_arg = rows_u32;
                    let mut cols_arg = cols_u32;
                    let mut arguments = [
                        (&mut packed_arg as *mut CuDevicePtr).cast::<c_void>(),
                        (&mut scales_arg as *mut CuDevicePtr).cast::<c_void>(),
                        (&mut input_arg as *mut CuDevicePtr).cast::<c_void>(),
                        (&mut output_arg as *mut CuDevicePtr).cast::<c_void>(),
                        (&mut rows_arg as *mut u32).cast::<c_void>(),
                        (&mut cols_arg as *mut u32).cast::<c_void>(),
                    ];
                    let threads = 128_u32;
                    check(
                        (self.api.launch_kernel)(
                            function,
                            rows_u32.div_ceil(threads),
                            1,
                            1,
                            threads,
                            1,
                            1,
                            0,
                            ptr::null_mut(),
                            arguments.as_mut_ptr(),
                            ptr::null_mut(),
                        ),
                        "cuLaunchKernel(INT4/BF16 group-32)",
                    )?;
                    check((self.api.ctx_synchronize)(), "cuCtxSynchronize")?;
                    let mut result = vec![0.0_f32; rows];
                    check(
                        (self.api.memcpy_dtoh)(
                            result.as_mut_ptr().cast(),
                            output,
                            rows * size_of::<f32>(),
                        ),
                        "cuMemcpyDtoH",
                    )?;
                    Ok(result)
                })();
                operation
            }
        }

        fn launch_int4_group32_bf16_matmul(
            &self,
            packed: CuDevicePtr,
            scales: CuDevicePtr,
            input: CuDevicePtr,
            output: CuDevicePtr,
            rows: usize,
            cols: usize,
            positions: usize,
        ) -> Result<Vec<f32>, String> {
            let rows_u32 =
                u32::try_from(rows).map_err(|_| "row count exceeds CUDA u32".to_string())?;
            let cols_u32 =
                u32::try_from(cols).map_err(|_| "column count exceeds CUDA u32".to_string())?;
            let positions_u32 = u32::try_from(positions)
                .map_err(|_| "position count exceeds CUDA u32".to_string())?;
            let output_elements = rows
                .checked_mul(positions)
                .ok_or_else(|| "INT4 batch output overflow".to_string())?;
            let output_u32 = u32::try_from(output_elements)
                .map_err(|_| "INT4 batch output exceeds CUDA u32".to_string())?;
            let function = self.kernel_function(
                "int4_group32_bf16_matmul",
                INT4_GROUP32_BF16_MATMUL_PTX,
                c"sytra_int4_group32_bf16_matmul".as_ptr(),
            )?;
            unsafe {
                check((self.api.ctx_set_current)(self.context), "cuCtxSetCurrent")?;
                let operation = (|| {
                    let mut packed_arg = packed;
                    let mut scales_arg = scales;
                    let mut input_arg = input;
                    let mut output_arg = output;
                    let mut rows_arg = rows_u32;
                    let mut cols_arg = cols_u32;
                    let mut positions_arg = positions_u32;
                    let mut arguments = [
                        (&mut packed_arg as *mut CuDevicePtr).cast::<c_void>(),
                        (&mut scales_arg as *mut CuDevicePtr).cast::<c_void>(),
                        (&mut input_arg as *mut CuDevicePtr).cast::<c_void>(),
                        (&mut output_arg as *mut CuDevicePtr).cast::<c_void>(),
                        (&mut rows_arg as *mut u32).cast::<c_void>(),
                        (&mut cols_arg as *mut u32).cast::<c_void>(),
                        (&mut positions_arg as *mut u32).cast::<c_void>(),
                    ];
                    let threads = 128_u32;
                    check(
                        (self.api.launch_kernel)(
                            function,
                            output_u32.div_ceil(threads),
                            1,
                            1,
                            threads,
                            1,
                            1,
                            0,
                            ptr::null_mut(),
                            arguments.as_mut_ptr(),
                            ptr::null_mut(),
                        ),
                        "cuLaunchKernel(INT4/BF16 batched matmul)",
                    )?;
                    check((self.api.ctx_synchronize)(), "cuCtxSynchronize")?;
                    let mut result = vec![0.0_f32; output_elements];
                    check(
                        (self.api.memcpy_dtoh)(
                            result.as_mut_ptr().cast(),
                            output,
                            output_elements * size_of::<f32>(),
                        ),
                        "cuMemcpyDtoH",
                    )?;
                    Ok(result)
                })();
                operation
            }
        }

        fn launch_bf16_matvec(
            &self,
            weights: CuDevicePtr,
            input: CuDevicePtr,
            output: CuDevicePtr,
            rows: usize,
            cols: usize,
        ) -> Result<Vec<f32>, String> {
            let rows_u32 =
                u32::try_from(rows).map_err(|_| "row count exceeds CUDA u32".to_string())?;
            let cols_u32 =
                u32::try_from(cols).map_err(|_| "column count exceeds CUDA u32".to_string())?;
            let function = self.kernel_function(
                "bf16_matvec",
                BF16_MATVEC_PTX,
                c"sytra_bf16_matvec".as_ptr(),
            )?;
            unsafe {
                check((self.api.ctx_set_current)(self.context), "cuCtxSetCurrent")?;
                let operation = (|| {
                    let mut weights_arg = weights;
                    let mut input_arg = input;
                    let mut output_arg = output;
                    let mut rows_arg = rows_u32;
                    let mut cols_arg = cols_u32;
                    let mut arguments = [
                        (&mut weights_arg as *mut CuDevicePtr).cast::<c_void>(),
                        (&mut input_arg as *mut CuDevicePtr).cast::<c_void>(),
                        (&mut output_arg as *mut CuDevicePtr).cast::<c_void>(),
                        (&mut rows_arg as *mut u32).cast::<c_void>(),
                        (&mut cols_arg as *mut u32).cast::<c_void>(),
                    ];
                    let threads = 128_u32;
                    check(
                        (self.api.launch_kernel)(
                            function,
                            rows_u32.div_ceil(threads),
                            1,
                            1,
                            threads,
                            1,
                            1,
                            0,
                            ptr::null_mut(),
                            arguments.as_mut_ptr(),
                            ptr::null_mut(),
                        ),
                        "cuLaunchKernel(BF16 matvec)",
                    )?;
                    check((self.api.ctx_synchronize)(), "cuCtxSynchronize")?;
                    let mut result = vec![0.0_f32; rows];
                    check(
                        (self.api.memcpy_dtoh)(
                            result.as_mut_ptr().cast(),
                            output,
                            rows * size_of::<f32>(),
                        ),
                        "cuMemcpyDtoH",
                    )?;
                    Ok(result)
                })();
                operation
            }
        }

        fn launch_bf16_transpose_matvec(
            &self,
            weights: CuDevicePtr,
            input: CuDevicePtr,
            output: CuDevicePtr,
            rows: usize,
            cols: usize,
        ) -> Result<Vec<f32>, String> {
            let rows_u32 =
                u32::try_from(rows).map_err(|_| "row count exceeds CUDA u32".to_string())?;
            let cols_u32 =
                u32::try_from(cols).map_err(|_| "column count exceeds CUDA u32".to_string())?;
            let function = self.kernel_function(
                "bf16_transpose_matvec",
                BF16_MATVEC_PTX,
                c"sytra_bf16_transpose_matvec".as_ptr(),
            )?;
            unsafe {
                check((self.api.ctx_set_current)(self.context), "cuCtxSetCurrent")?;
                let operation = (|| {
                    let mut weights_arg = weights;
                    let mut input_arg = input;
                    let mut output_arg = output;
                    let mut rows_arg = rows_u32;
                    let mut cols_arg = cols_u32;
                    let mut arguments = [
                        (&mut weights_arg as *mut CuDevicePtr).cast::<c_void>(),
                        (&mut input_arg as *mut CuDevicePtr).cast::<c_void>(),
                        (&mut output_arg as *mut CuDevicePtr).cast::<c_void>(),
                        (&mut rows_arg as *mut u32).cast::<c_void>(),
                        (&mut cols_arg as *mut u32).cast::<c_void>(),
                    ];
                    let threads = 128_u32;
                    check(
                        (self.api.launch_kernel)(
                            function,
                            cols_u32.div_ceil(threads),
                            1,
                            1,
                            threads,
                            1,
                            1,
                            0,
                            ptr::null_mut(),
                            arguments.as_mut_ptr(),
                            ptr::null_mut(),
                        ),
                        "cuLaunchKernel(BF16 transpose matvec)",
                    )?;
                    check((self.api.ctx_synchronize)(), "cuCtxSynchronize")?;
                    let mut result = vec![0.0_f32; cols];
                    check(
                        (self.api.memcpy_dtoh)(
                            result.as_mut_ptr().cast(),
                            output,
                            cols * size_of::<f32>(),
                        ),
                        "cuMemcpyDtoH",
                    )?;
                    Ok(result)
                })();
                operation
            }
        }

        fn launch_bf16_matmul(
            &self,
            weights: CuDevicePtr,
            input: CuDevicePtr,
            output: CuDevicePtr,
            rows: usize,
            cols: usize,
            positions: usize,
        ) -> Result<Vec<f32>, String> {
            let rows_u32 =
                u32::try_from(rows).map_err(|_| "row count exceeds CUDA u32".to_string())?;
            let cols_u32 =
                u32::try_from(cols).map_err(|_| "column count exceeds CUDA u32".to_string())?;
            let positions_u32 = u32::try_from(positions)
                .map_err(|_| "position count exceeds CUDA u32".to_string())?;
            let outputs = rows_u32
                .checked_mul(positions_u32)
                .ok_or_else(|| "CUDA batch output count exceeds u32".to_string())?;
            let function = self.kernel_function(
                "bf16_matmul",
                BF16_MATVEC_PTX,
                c"sytra_bf16_matmul".as_ptr(),
            )?;
            unsafe {
                check((self.api.ctx_set_current)(self.context), "cuCtxSetCurrent")?;
                let mut weights_arg = weights;
                let mut input_arg = input;
                let mut output_arg = output;
                let mut rows_arg = rows_u32;
                let mut cols_arg = cols_u32;
                let mut positions_arg = positions_u32;
                let mut arguments = [
                    (&mut weights_arg as *mut CuDevicePtr).cast::<c_void>(),
                    (&mut input_arg as *mut CuDevicePtr).cast::<c_void>(),
                    (&mut output_arg as *mut CuDevicePtr).cast::<c_void>(),
                    (&mut rows_arg as *mut u32).cast::<c_void>(),
                    (&mut cols_arg as *mut u32).cast::<c_void>(),
                    (&mut positions_arg as *mut u32).cast::<c_void>(),
                ];
                let threads = 128_u32;
                check(
                    (self.api.launch_kernel)(
                        function,
                        outputs.div_ceil(threads),
                        1,
                        1,
                        threads,
                        1,
                        1,
                        0,
                        ptr::null_mut(),
                        arguments.as_mut_ptr(),
                        ptr::null_mut(),
                    ),
                    "cuLaunchKernel(BF16 matmul)",
                )?;
                check((self.api.ctx_synchronize)(), "cuCtxSynchronize")?;
                let output_elements = rows
                    .checked_mul(positions)
                    .ok_or_else(|| "CUDA batch output size overflow".to_string())?;
                let mut result = vec![0.0_f32; output_elements];
                check(
                    (self.api.memcpy_dtoh)(
                        result.as_mut_ptr().cast(),
                        output,
                        output_elements * size_of::<f32>(),
                    ),
                    "cuMemcpyDtoH",
                )?;
                Ok(result)
            }
        }

        fn launch_bf16_transpose_matmul(
            &self,
            weights: CuDevicePtr,
            input: CuDevicePtr,
            output: CuDevicePtr,
            rows: usize,
            cols: usize,
            positions: usize,
        ) -> Result<Vec<f32>, String> {
            let rows_u32 =
                u32::try_from(rows).map_err(|_| "row count exceeds CUDA u32".to_string())?;
            let cols_u32 =
                u32::try_from(cols).map_err(|_| "column count exceeds CUDA u32".to_string())?;
            let positions_u32 = u32::try_from(positions)
                .map_err(|_| "position count exceeds CUDA u32".to_string())?;
            let outputs = cols_u32
                .checked_mul(positions_u32)
                .ok_or_else(|| "CUDA transpose batch output exceeds u32".to_string())?;
            let function = self.kernel_function(
                "bf16_transpose_matmul",
                BF16_MATVEC_PTX,
                c"sytra_bf16_transpose_matmul".as_ptr(),
            )?;
            unsafe {
                check((self.api.ctx_set_current)(self.context), "cuCtxSetCurrent")?;
                let mut weights_arg = weights;
                let mut input_arg = input;
                let mut output_arg = output;
                let mut rows_arg = rows_u32;
                let mut cols_arg = cols_u32;
                let mut positions_arg = positions_u32;
                let mut arguments = [
                    (&mut weights_arg as *mut CuDevicePtr).cast::<c_void>(),
                    (&mut input_arg as *mut CuDevicePtr).cast::<c_void>(),
                    (&mut output_arg as *mut CuDevicePtr).cast::<c_void>(),
                    (&mut rows_arg as *mut u32).cast::<c_void>(),
                    (&mut cols_arg as *mut u32).cast::<c_void>(),
                    (&mut positions_arg as *mut u32).cast::<c_void>(),
                ];
                let threads = 128_u32;
                check(
                    (self.api.launch_kernel)(
                        function,
                        outputs.div_ceil(threads),
                        1,
                        1,
                        threads,
                        1,
                        1,
                        0,
                        ptr::null_mut(),
                        arguments.as_mut_ptr(),
                        ptr::null_mut(),
                    ),
                    "cuLaunchKernel(BF16 transpose matmul)",
                )?;
                check((self.api.ctx_synchronize)(), "cuCtxSynchronize")?;
                let output_elements = cols
                    .checked_mul(positions)
                    .ok_or_else(|| "CUDA transpose batch output overflow".to_string())?;
                let mut result = vec![0.0_f32; output_elements];
                check(
                    (self.api.memcpy_dtoh)(
                        result.as_mut_ptr().cast(),
                        output,
                        output_elements * size_of::<f32>(),
                    ),
                    "cuMemcpyDtoH",
                )?;
                Ok(result)
            }
        }

        #[allow(clippy::too_many_arguments)]
        fn launch_standard_attention_bf16(
            &self,
            keys: CuDevicePtr,
            values: CuDevicePtr,
            query: CuDevicePtr,
            output: CuDevicePtr,
            cache_len: usize,
            heads: usize,
            kv_heads: usize,
            head_dim: usize,
            value_dim: usize,
            softmax_scale: f32,
            window: usize,
        ) -> Result<Vec<f32>, String> {
            let to_u32 = |value: usize, name: &str| {
                u32::try_from(value).map_err(|_| format!("{name} exceeds CUDA u32"))
            };
            let cache_len_u32 = to_u32(cache_len, "KV cache length")?;
            let heads_u32 = to_u32(heads, "attention head count")?;
            let kv_heads_u32 = to_u32(kv_heads, "KV head count")?;
            let head_dim_u32 = to_u32(head_dim, "attention head dimension")?;
            let value_dim_u32 = to_u32(value_dim, "attention value dimension")?;
            let window_u32 = to_u32(window, "attention window")?;
            let outputs = heads_u32
                .checked_mul(value_dim_u32)
                .ok_or_else(|| "CUDA attention output exceeds u32".to_string())?;
            let function = self.kernel_function(
                "standard_attention_bf16",
                STANDARD_ATTENTION_BF16_PTX,
                c"sytra_standard_attention_bf16".as_ptr(),
            )?;
            unsafe {
                check((self.api.ctx_set_current)(self.context), "cuCtxSetCurrent")?;
                let mut keys_arg = keys;
                let mut values_arg = values;
                let mut query_arg = query;
                let mut output_arg = output;
                let mut cache_len_arg = cache_len_u32;
                let mut heads_arg = heads_u32;
                let mut kv_heads_arg = kv_heads_u32;
                let mut head_dim_arg = head_dim_u32;
                let mut value_dim_arg = value_dim_u32;
                let mut scale_arg = softmax_scale;
                let mut window_arg = window_u32;
                let mut arguments = [
                    (&mut keys_arg as *mut CuDevicePtr).cast::<c_void>(),
                    (&mut values_arg as *mut CuDevicePtr).cast::<c_void>(),
                    (&mut query_arg as *mut CuDevicePtr).cast::<c_void>(),
                    (&mut output_arg as *mut CuDevicePtr).cast::<c_void>(),
                    (&mut cache_len_arg as *mut u32).cast::<c_void>(),
                    (&mut heads_arg as *mut u32).cast::<c_void>(),
                    (&mut kv_heads_arg as *mut u32).cast::<c_void>(),
                    (&mut head_dim_arg as *mut u32).cast::<c_void>(),
                    (&mut value_dim_arg as *mut u32).cast::<c_void>(),
                    (&mut scale_arg as *mut f32).cast::<c_void>(),
                    (&mut window_arg as *mut u32).cast::<c_void>(),
                ];
                let threads = 128_u32;
                check(
                    (self.api.launch_kernel)(
                        function,
                        outputs.div_ceil(threads),
                        1,
                        1,
                        threads,
                        1,
                        1,
                        0,
                        ptr::null_mut(),
                        arguments.as_mut_ptr(),
                        ptr::null_mut(),
                    ),
                    "cuLaunchKernel(standard BF16 attention)",
                )?;
                check((self.api.ctx_synchronize)(), "cuCtxSynchronize")?;
                let mut result = vec![0.0_f32; heads * value_dim];
                check(
                    (self.api.memcpy_dtoh)(
                        result.as_mut_ptr().cast(),
                        output,
                        result.len() * size_of::<f32>(),
                    ),
                    "cuMemcpyDtoH",
                )?;
                Ok(result)
            }
        }
    }

    fn as_bytes<T>(values: &[T]) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
        }
    }

    // One thread computes one `[head, value_dim]` output. Keys and values stay
    // BF16 in their persistent request allocation. Logits are recomputed in a
    // stable two-pass softmax so no context-sized scratch buffer is required.
    const STANDARD_ATTENTION_BF16_PTX: &[u8] = br#"
.version 7.1
.target sm_70
.address_size 64

.visible .entry sytra_standard_attention_bf16(
    .param .u64 p_keys,
    .param .u64 p_values,
    .param .u64 p_query,
    .param .u64 p_output,
    .param .u32 p_cache_len,
    .param .u32 p_heads,
    .param .u32 p_kv_heads,
    .param .u32 p_head_dim,
    .param .u32 p_value_dim,
    .param .f32 p_scale,
    .param .u32 p_window
)
{
    .reg .pred %p<6>;
    .reg .b16 %rs<3>;
    .reg .b32 %r<48>;
    .reg .b64 %rd<20>;
    .reg .f32 %f<16>;

    ld.param.u64 %rd1, [p_keys];
    ld.param.u64 %rd2, [p_values];
    ld.param.u64 %rd3, [p_query];
    ld.param.u64 %rd4, [p_output];
    ld.param.u32 %r1, [p_cache_len];
    ld.param.u32 %r2, [p_heads];
    ld.param.u32 %r3, [p_kv_heads];
    ld.param.u32 %r4, [p_head_dim];
    ld.param.u32 %r5, [p_value_dim];
    ld.param.f32 %f1, [p_scale];
    ld.param.u32 %r6, [p_window];

    mov.u32 %r7, %ctaid.x;
    mov.u32 %r8, %ntid.x;
    mov.u32 %r9, %tid.x;
    mad.lo.s32 %r10, %r7, %r8, %r9;
    mul.lo.u32 %r11, %r2, %r5;
    setp.ge.u32 %p1, %r10, %r11;
    @%p1 bra A_DONE;

    div.u32 %r12, %r10, %r5;
    rem.u32 %r13, %r10, %r5;
    div.u32 %r14, %r2, %r3;
    div.u32 %r15, %r12, %r14;
    mov.u32 %r16, 0;
    setp.eq.u32 %p2, %r6, 0;
    @%p2 bra A_START_READY;
    setp.le.u32 %p3, %r1, %r6;
    @%p3 bra A_START_READY;
    sub.u32 %r16, %r1, %r6;

A_START_READY:
    mov.f32 %f2, 0fFF800000;
    mov.u32 %r17, %r16;

A_MAX_POS:
    setp.ge.u32 %p1, %r17, %r1;
    @%p1 bra A_MAX_DONE;
    mov.f32 %f3, 0f00000000;
    mov.u32 %r18, 0;

A_MAX_DOT:
    setp.ge.u32 %p1, %r18, %r4;
    @%p1 bra A_MAX_ACCUM;
    mul.lo.u32 %r19, %r12, %r4;
    add.u32 %r20, %r19, %r18;
    mul.wide.u32 %rd5, %r20, 4;
    add.s64 %rd6, %rd3, %rd5;
    ld.global.f32 %f4, [%rd6];
    mul.lo.u32 %r21, %r17, %r3;
    add.u32 %r22, %r21, %r15;
    mul.lo.u32 %r23, %r22, %r4;
    add.u32 %r24, %r23, %r18;
    mul.wide.u32 %rd7, %r24, 2;
    add.s64 %rd8, %rd1, %rd7;
    ld.global.u16 %rs1, [%rd8];
    cvt.u32.u16 %r25, %rs1;
    shl.b32 %r26, %r25, 16;
    mov.b32 %f5, %r26;
    fma.rn.f32 %f3, %f4, %f5, %f3;
    add.u32 %r18, %r18, 1;
    bra A_MAX_DOT;

A_MAX_ACCUM:
    mul.f32 %f3, %f3, %f1;
    max.f32 %f2, %f2, %f3;
    add.u32 %r17, %r17, 1;
    bra A_MAX_POS;

A_MAX_DONE:
    mov.f32 %f6, 0f00000000;
    mov.f32 %f7, 0f00000000;
    mov.u32 %r17, %r16;

A_SUM_POS:
    setp.ge.u32 %p1, %r17, %r1;
    @%p1 bra A_STORE;
    mov.f32 %f3, 0f00000000;
    mov.u32 %r18, 0;

A_SUM_DOT:
    setp.ge.u32 %p1, %r18, %r4;
    @%p1 bra A_SUM_ACCUM;
    mul.lo.u32 %r19, %r12, %r4;
    add.u32 %r20, %r19, %r18;
    mul.wide.u32 %rd5, %r20, 4;
    add.s64 %rd6, %rd3, %rd5;
    ld.global.f32 %f4, [%rd6];
    mul.lo.u32 %r21, %r17, %r3;
    add.u32 %r22, %r21, %r15;
    mul.lo.u32 %r23, %r22, %r4;
    add.u32 %r24, %r23, %r18;
    mul.wide.u32 %rd7, %r24, 2;
    add.s64 %rd8, %rd1, %rd7;
    ld.global.u16 %rs1, [%rd8];
    cvt.u32.u16 %r25, %rs1;
    shl.b32 %r26, %r25, 16;
    mov.b32 %f5, %r26;
    fma.rn.f32 %f3, %f4, %f5, %f3;
    add.u32 %r18, %r18, 1;
    bra A_SUM_DOT;

A_SUM_ACCUM:
    mul.f32 %f3, %f3, %f1;
    sub.f32 %f8, %f3, %f2;
    mul.f32 %f8, %f8, 0f3FB8AA3B;
    ex2.approx.f32 %f9, %f8;
    add.f32 %f6, %f6, %f9;
    mul.lo.u32 %r27, %r17, %r3;
    add.u32 %r28, %r27, %r15;
    mul.lo.u32 %r29, %r28, %r5;
    add.u32 %r30, %r29, %r13;
    mul.wide.u32 %rd9, %r30, 2;
    add.s64 %rd10, %rd2, %rd9;
    ld.global.u16 %rs2, [%rd10];
    cvt.u32.u16 %r31, %rs2;
    shl.b32 %r32, %r31, 16;
    mov.b32 %f10, %r32;
    fma.rn.f32 %f7, %f9, %f10, %f7;
    add.u32 %r17, %r17, 1;
    bra A_SUM_POS;

A_STORE:
    div.rn.f32 %f11, %f7, %f6;
    mul.wide.u32 %rd11, %r10, 4;
    add.s64 %rd12, %rd4, %rd11;
    st.global.f32 [%rd12], %f11;

A_DONE:
    ret;
}
"#;

    // One thread computes one output row. This is a portable correctness
    // kernel, not the eventual tensor-core implementation.
    const INT4_GROUP32_PTX: &[u8] = br#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry sytra_int4_group32_matvec(
    .param .u64 p_packed,
    .param .u64 p_scales,
    .param .u64 p_input,
    .param .u64 p_output,
    .param .u32 p_rows,
    .param .u32 p_cols
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<24>;
    .reg .b64 %rd<16>;
    .reg .f32 %f<8>;

    ld.param.u64 %rd1, [p_packed];
    ld.param.u64 %rd2, [p_scales];
    ld.param.u64 %rd3, [p_input];
    ld.param.u64 %rd4, [p_output];
    ld.param.u32 %r1, [p_rows];
    ld.param.u32 %r2, [p_cols];

    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.s32 %r6, %r3, %r4, %r5;
    setp.ge.u32 %p1, %r6, %r1;
    @%p1 bra DONE;

    mov.f32 %f1, 0f00000000;
    mov.u32 %r7, 0;
    div.u32 %r8, %r2, 8;
    div.u32 %r9, %r2, 32;

LOOP:
    setp.ge.u32 %p1, %r7, %r2;
    @%p1 bra STORE;

    mul.lo.u32 %r10, %r6, %r8;
    div.u32 %r11, %r7, 8;
    add.u32 %r12, %r10, %r11;
    mul.wide.u32 %rd5, %r12, 4;
    add.s64 %rd6, %rd1, %rd5;
    ld.global.u32 %r13, [%rd6];

    and.b32 %r14, %r7, 7;
    shl.b32 %r15, %r14, 2;
    shr.u32 %r16, %r13, %r15;
    and.b32 %r17, %r16, 15;
    sub.s32 %r18, %r17, 8;
    cvt.rn.f32.s32 %f2, %r18;

    mul.lo.u32 %r19, %r6, %r9;
    div.u32 %r20, %r7, 32;
    add.u32 %r21, %r19, %r20;
    mul.wide.u32 %rd7, %r21, 4;
    add.s64 %rd8, %rd2, %rd7;
    ld.global.f32 %f3, [%rd8];

    mul.wide.u32 %rd9, %r7, 4;
    add.s64 %rd10, %rd3, %rd9;
    ld.global.f32 %f4, [%rd10];
    mul.f32 %f5, %f2, %f3;
    fma.rn.f32 %f1, %f5, %f4, %f1;

    add.u32 %r7, %r7, 1;
    bra LOOP;

STORE:
    mul.wide.u32 %rd11, %r6, 4;
    add.s64 %rd12, %rd4, %rd11;
    st.global.f32 [%rd12], %f1;
DONE:
    ret;
}

"#;

    const INT4_GROUP32_BF16_PTX: &[u8] = br#"
.version 7.1
.target sm_70
.address_size 64

.visible .entry sytra_int4_group32_bf16_matvec(
    .param .u64 p_packed,
    .param .u64 p_scales,
    .param .u64 p_input,
    .param .u64 p_output,
    .param .u32 p_rows,
    .param .u32 p_cols
)
{
    .reg .pred %p<2>;
    .reg .b16 %rs<2>;
    .reg .b32 %r<24>;
    .reg .b64 %rd<16>;
    .reg .f32 %f<8>;

    ld.param.u64 %rd1, [p_packed];
    ld.param.u64 %rd2, [p_scales];
    ld.param.u64 %rd3, [p_input];
    ld.param.u64 %rd4, [p_output];
    ld.param.u32 %r1, [p_rows];
    ld.param.u32 %r2, [p_cols];
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.s32 %r6, %r3, %r4, %r5;
    setp.ge.u32 %p1, %r6, %r1;
    @%p1 bra DONE_BF16;
    mov.f32 %f1, 0f00000000;
    mov.u32 %r7, 0;
    div.u32 %r8, %r2, 8;
    div.u32 %r9, %r2, 32;

LOOP_BF16:
    setp.ge.u32 %p1, %r7, %r2;
    @%p1 bra STORE_BF16;
    mul.lo.u32 %r10, %r6, %r8;
    div.u32 %r11, %r7, 8;
    add.u32 %r12, %r10, %r11;
    mul.wide.u32 %rd5, %r12, 4;
    add.s64 %rd6, %rd1, %rd5;
    ld.global.u32 %r13, [%rd6];
    and.b32 %r14, %r7, 7;
    shl.b32 %r15, %r14, 2;
    shr.u32 %r16, %r13, %r15;
    and.b32 %r17, %r16, 15;
    sub.s32 %r18, %r17, 8;
    cvt.rn.f32.s32 %f2, %r18;
    mul.lo.u32 %r19, %r6, %r9;
    div.u32 %r20, %r7, 32;
    add.u32 %r21, %r19, %r20;
    mul.wide.u32 %rd7, %r21, 2;
    add.s64 %rd8, %rd2, %rd7;
    ld.global.u16 %rs1, [%rd8];
    cvt.u32.u16 %r22, %rs1;
    shl.b32 %r23, %r22, 16;
    mov.b32 %f3, %r23;
    mul.wide.u32 %rd9, %r7, 4;
    add.s64 %rd10, %rd3, %rd9;
    ld.global.f32 %f4, [%rd10];
    mul.f32 %f5, %f2, %f3;
    fma.rn.f32 %f1, %f5, %f4, %f1;
    add.u32 %r7, %r7, 1;
    bra LOOP_BF16;

STORE_BF16:
    mul.wide.u32 %rd11, %r6, 4;
    add.s64 %rd12, %rd4, %rd11;
    st.global.f32 [%rd12], %f1;
DONE_BF16:
    ret;
}
"#;

    // One thread computes one (position, output-row) pair. Weight rows and
    // BF16 scales are shared by every position in the launch.
    const INT4_GROUP32_BF16_MATMUL_PTX: &[u8] = br#"
.version 7.1
.target sm_70
.address_size 64

.visible .entry sytra_int4_group32_bf16_matmul(
    .param .u64 p_packed,
    .param .u64 p_scales,
    .param .u64 p_input,
    .param .u64 p_output,
    .param .u32 p_rows,
    .param .u32 p_cols,
    .param .u32 p_positions
)
{
    .reg .pred %p<2>;
    .reg .b16 %rs<2>;
    .reg .b32 %r<32>;
    .reg .b64 %rd<16>;
    .reg .f32 %f<8>;

    ld.param.u64 %rd1, [p_packed];
    ld.param.u64 %rd2, [p_scales];
    ld.param.u64 %rd3, [p_input];
    ld.param.u64 %rd4, [p_output];
    ld.param.u32 %r1, [p_rows];
    ld.param.u32 %r2, [p_cols];
    ld.param.u32 %r3, [p_positions];
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mov.u32 %r6, %tid.x;
    mad.lo.s32 %r7, %r4, %r5, %r6;
    mul.lo.u32 %r8, %r1, %r3;
    setp.ge.u32 %p1, %r7, %r8;
    @%p1 bra DONE_BATCH;
    div.u32 %r9, %r7, %r1;
    rem.u32 %r10, %r7, %r1;
    mov.f32 %f1, 0f00000000;
    mov.u32 %r11, 0;
    div.u32 %r12, %r2, 8;
    div.u32 %r13, %r2, 32;

LOOP_BATCH:
    setp.ge.u32 %p1, %r11, %r2;
    @%p1 bra STORE_BATCH;
    mul.lo.u32 %r14, %r10, %r12;
    div.u32 %r15, %r11, 8;
    add.u32 %r16, %r14, %r15;
    mul.wide.u32 %rd5, %r16, 4;
    add.s64 %rd6, %rd1, %rd5;
    ld.global.u32 %r17, [%rd6];
    and.b32 %r18, %r11, 7;
    shl.b32 %r19, %r18, 2;
    shr.u32 %r20, %r17, %r19;
    and.b32 %r21, %r20, 15;
    sub.s32 %r22, %r21, 8;
    cvt.rn.f32.s32 %f2, %r22;
    mul.lo.u32 %r23, %r10, %r13;
    div.u32 %r24, %r11, 32;
    add.u32 %r25, %r23, %r24;
    mul.wide.u32 %rd7, %r25, 2;
    add.s64 %rd8, %rd2, %rd7;
    ld.global.u16 %rs1, [%rd8];
    cvt.u32.u16 %r28, %rs1;
    shl.b32 %r29, %r28, 16;
    mov.b32 %f3, %r29;
    mul.lo.u32 %r26, %r9, %r2;
    add.u32 %r27, %r26, %r11;
    mul.wide.u32 %rd9, %r27, 4;
    add.s64 %rd10, %rd3, %rd9;
    ld.global.f32 %f4, [%rd10];
    mul.f32 %f5, %f2, %f3;
    fma.rn.f32 %f1, %f5, %f4, %f1;
    add.u32 %r11, %r11, 1;
    bra LOOP_BATCH;

STORE_BATCH:
    mul.wide.u32 %rd11, %r7, 4;
    add.s64 %rd12, %rd4, %rd11;
    st.global.f32 [%rd12], %f1;
DONE_BATCH:
    ret;
}
"#;

    const BF16_MATVEC_PTX: &[u8] = br#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry sytra_bf16_matvec(
    .param .u64 p_weights,
    .param .u64 p_input,
    .param .u64 p_output,
    .param .u32 p_rows,
    .param .u32 p_cols
)
{
    .reg .pred %p<2>;
    .reg .b16 %rs<2>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<16>;
    .reg .f32 %f<5>;

    ld.param.u64 %rd1, [p_weights];
    ld.param.u64 %rd2, [p_input];
    ld.param.u64 %rd3, [p_output];
    ld.param.u32 %r1, [p_rows];
    ld.param.u32 %r2, [p_cols];
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.s32 %r6, %r3, %r4, %r5;
    setp.ge.u32 %p1, %r6, %r1;
    @%p1 bra DONE;
    mov.f32 %f1, 0f00000000;
    mov.u32 %r7, 0;

LOOP:
    setp.ge.u32 %p1, %r7, %r2;
    @%p1 bra STORE;
    mul.lo.u32 %r8, %r6, %r2;
    add.u32 %r9, %r8, %r7;
    mul.wide.u32 %rd4, %r9, 2;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.u16 %rs1, [%rd5];
    cvt.u32.u16 %r10, %rs1;
    shl.b32 %r11, %r10, 16;
    mov.b32 %f2, %r11;
    mul.wide.u32 %rd6, %r7, 4;
    add.s64 %rd7, %rd2, %rd6;
    ld.global.f32 %f3, [%rd7];
    fma.rn.f32 %f1, %f2, %f3, %f1;
    add.u32 %r7, %r7, 1;
    bra LOOP;

STORE:
    mul.wide.u32 %rd8, %r6, 4;
    add.s64 %rd9, %rd3, %rd8;
    st.global.f32 [%rd9], %f1;
DONE:
    ret;
}

.visible .entry sytra_bf16_transpose_matvec(
    .param .u64 p_weights,
    .param .u64 p_input,
    .param .u64 p_output,
    .param .u32 p_rows,
    .param .u32 p_cols
)
{
    .reg .pred %p<2>;
    .reg .b16 %rs<2>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<16>;
    .reg .f32 %f<5>;

    ld.param.u64 %rd1, [p_weights];
    ld.param.u64 %rd2, [p_input];
    ld.param.u64 %rd3, [p_output];
    ld.param.u32 %r1, [p_rows];
    ld.param.u32 %r2, [p_cols];
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.s32 %r6, %r3, %r4, %r5;
    setp.ge.u32 %p1, %r6, %r2;
    @%p1 bra T_DONE;
    mov.f32 %f1, 0f00000000;
    mov.u32 %r7, 0;

T_LOOP:
    setp.ge.u32 %p1, %r7, %r1;
    @%p1 bra T_STORE;
    mul.lo.u32 %r8, %r7, %r2;
    add.u32 %r9, %r8, %r6;
    mul.wide.u32 %rd4, %r9, 2;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.u16 %rs1, [%rd5];
    cvt.u32.u16 %r10, %rs1;
    shl.b32 %r11, %r10, 16;
    mov.b32 %f2, %r11;
    mul.wide.u32 %rd6, %r7, 4;
    add.s64 %rd7, %rd2, %rd6;
    ld.global.f32 %f3, [%rd7];
    fma.rn.f32 %f1, %f2, %f3, %f1;
    add.u32 %r7, %r7, 1;
    bra T_LOOP;

T_STORE:
    mul.wide.u32 %rd8, %r6, 4;
    add.s64 %rd9, %rd3, %rd8;
    st.global.f32 [%rd9], %f1;
T_DONE:
    ret;
}

.visible .entry sytra_bf16_matmul(
    .param .u64 p_weights,
    .param .u64 p_input,
    .param .u64 p_output,
    .param .u32 p_rows,
    .param .u32 p_cols,
    .param .u32 p_positions
)
{
    .reg .pred %p<2>;
    .reg .b16 %rs<2>;
    .reg .b32 %r<20>;
    .reg .b64 %rd<16>;
    .reg .f32 %f<5>;

    ld.param.u64 %rd1, [p_weights];
    ld.param.u64 %rd2, [p_input];
    ld.param.u64 %rd3, [p_output];
    ld.param.u32 %r1, [p_rows];
    ld.param.u32 %r2, [p_cols];
    ld.param.u32 %r3, [p_positions];
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mov.u32 %r6, %tid.x;
    mad.lo.s32 %r7, %r4, %r5, %r6;
    mul.lo.u32 %r8, %r1, %r3;
    setp.ge.u32 %p1, %r7, %r8;
    @%p1 bra M_DONE;
    div.u32 %r9, %r7, %r1;
    rem.u32 %r10, %r7, %r1;
    mov.f32 %f1, 0f00000000;
    mov.u32 %r11, 0;

M_LOOP:
    setp.ge.u32 %p1, %r11, %r2;
    @%p1 bra M_STORE;
    mul.lo.u32 %r12, %r10, %r2;
    add.u32 %r13, %r12, %r11;
    mul.wide.u32 %rd4, %r13, 2;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.u16 %rs1, [%rd5];
    cvt.u32.u16 %r14, %rs1;
    shl.b32 %r15, %r14, 16;
    mov.b32 %f2, %r15;
    mul.lo.u32 %r16, %r9, %r2;
    add.u32 %r17, %r16, %r11;
    mul.wide.u32 %rd6, %r17, 4;
    add.s64 %rd7, %rd2, %rd6;
    ld.global.f32 %f3, [%rd7];
    fma.rn.f32 %f1, %f2, %f3, %f1;
    add.u32 %r11, %r11, 1;
    bra M_LOOP;

M_STORE:
    mul.wide.u32 %rd8, %r7, 4;
    add.s64 %rd9, %rd3, %rd8;
    st.global.f32 [%rd9], %f1;
M_DONE:
    ret;
}

.visible .entry sytra_bf16_transpose_matmul(
    .param .u64 p_weights,
    .param .u64 p_input,
    .param .u64 p_output,
    .param .u32 p_rows,
    .param .u32 p_cols,
    .param .u32 p_positions
)
{
    .reg .pred %p<2>;
    .reg .b16 %rs<2>;
    .reg .b32 %r<20>;
    .reg .b64 %rd<16>;
    .reg .f32 %f<5>;

    ld.param.u64 %rd1, [p_weights];
    ld.param.u64 %rd2, [p_input];
    ld.param.u64 %rd3, [p_output];
    ld.param.u32 %r1, [p_rows];
    ld.param.u32 %r2, [p_cols];
    ld.param.u32 %r3, [p_positions];
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mov.u32 %r6, %tid.x;
    mad.lo.s32 %r7, %r4, %r5, %r6;
    mul.lo.u32 %r8, %r2, %r3;
    setp.ge.u32 %p1, %r7, %r8;
    @%p1 bra TM_DONE;
    div.u32 %r9, %r7, %r2;
    rem.u32 %r10, %r7, %r2;
    mov.f32 %f1, 0f00000000;
    mov.u32 %r11, 0;

TM_LOOP:
    setp.ge.u32 %p1, %r11, %r1;
    @%p1 bra TM_STORE;
    mul.lo.u32 %r12, %r11, %r2;
    add.u32 %r13, %r12, %r10;
    mul.wide.u32 %rd4, %r13, 2;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.u16 %rs1, [%rd5];
    cvt.u32.u16 %r14, %rs1;
    shl.b32 %r15, %r14, 16;
    mov.b32 %f2, %r15;
    mul.lo.u32 %r16, %r9, %r1;
    add.u32 %r17, %r16, %r11;
    mul.wide.u32 %rd6, %r17, 4;
    add.s64 %rd7, %rd2, %rd6;
    ld.global.f32 %f3, [%rd7];
    fma.rn.f32 %f1, %f2, %f3, %f1;
    add.u32 %r11, %r11, 1;
    bra TM_LOOP;

TM_STORE:
    mul.wide.u32 %rd8, %r7, 4;
    add.s64 %rd9, %rd3, %rd8;
    st.global.f32 [%rd9], %f1;
TM_DONE:
    ret;
}
"#;

    impl Accelerator for CudaAccelerator {
        fn name(&self) -> &str {
            "cuda-driver"
        }

        fn upload(&self, _key: ExpertKey, bytes: &[u8]) -> Result<AcceleratorBuffer, String> {
            if bytes.is_empty() {
                return Err("cannot upload an empty expert".into());
            }
            let size = u64::try_from(bytes.len())
                .map_err(|_| "CUDA allocation size exceeds u64".to_string())?;
            if !reserve_global_bytes(&self.live_bytes, &self.peak_bytes, size, self.memory_budget) {
                self.budget_denials.fetch_add(1, Ordering::AcqRel);
                return Err(format!(
                    "CUDA allocation needs {size} bytes but the hard device limit is {}",
                    self.memory_budget
                ));
            }
            let mut pointer = 0;
            let upload = (|| unsafe {
                check((self.api.ctx_set_current)(self.context), "cuCtxSetCurrent")?;
                check(
                    (self.api.mem_alloc)(&mut pointer, bytes.len()),
                    "cuMemAlloc",
                )?;
                if let Err(error) = check(
                    (self.api.memcpy_htod)(pointer, bytes.as_ptr().cast::<c_void>(), bytes.len()),
                    "cuMemcpyHtoD",
                ) {
                    let _ = (self.api.mem_free)(pointer);
                    return Err(error);
                }
                Ok::<(), String>(())
            })();
            if let Err(error) = upload {
                self.live_bytes.fetch_sub(size, Ordering::AcqRel);
                return Err(error);
            }
            let mut allocations = match self.allocations.lock() {
                Ok(allocations) => allocations,
                Err(_) => {
                    unsafe {
                        let _ = (self.api.mem_free)(pointer);
                    }
                    self.live_bytes.fetch_sub(size, Ordering::AcqRel);
                    return Err("CUDA allocation registry is poisoned".into());
                }
            };
            allocations.insert(pointer, size);
            Ok(AcceleratorBuffer {
                id: pointer,
                bytes: bytes.len(),
            })
        }

        fn release(&self, buffer: &AcceleratorBuffer) {
            let allocation = self
                .allocations
                .lock()
                .map(|mut values| values.remove(&buffer.id))
                .unwrap_or(None);
            if let Some(size) = allocation {
                unsafe {
                    let _ = (self.api.ctx_set_current)(self.context);
                    let _ = (self.api.mem_free)(buffer.id);
                }
                self.live_bytes.fetch_sub(size, Ordering::AcqRel);
            }
        }
    }

    impl Drop for CudaAccelerator {
        fn drop(&mut self) {
            if let Ok(mut allocations) = self.allocations.lock() {
                unsafe {
                    let _ = (self.api.ctx_set_current)(self.context);
                    for (pointer, _) in allocations.drain() {
                        let _ = (self.api.mem_free)(pointer);
                    }
                    if let Ok(mut kernels) = self.kernels.lock() {
                        for (_, (module, _)) in kernels.drain() {
                            let _ = (self.api.module_unload)(module);
                        }
                    }
                    let _ = (self.api.primary_release)(self.device);
                }
            }
        }
    }

    fn check(result: CuResult, operation: &str) -> Result<(), String> {
        if result == 0 {
            Ok(())
        } else {
            Err(format!("{operation} failed with CUDA status {result}"))
        }
    }

    fn reserve_global_bytes(live: &AtomicU64, peak: &AtomicU64, size: u64, budget: u64) -> bool {
        let mut current = live.load(Ordering::Acquire);
        loop {
            let Some(next) = current.checked_add(size) else {
                return false;
            };
            if next > budget {
                return false;
            }
            match live.compare_exchange_weak(current, next, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => {
                    peak.fetch_max(next, Ordering::AcqRel);
                    return true;
                }
                Err(actual) => current = actual,
            }
        }
    }
}

#[cfg(not(target_os = "windows"))]
mod platform {
    use super::*;

    #[derive(Debug)]
    pub struct CudaAccelerator;

    impl CudaAccelerator {
        pub fn new(_ordinal: i32) -> Result<Self, String> {
            Err(
                "the dependency-free CUDA driver loader is currently implemented for Windows"
                    .into(),
            )
        }

        pub fn new_with_budget(_ordinal: i32, _memory_budget: u64) -> Result<Self, String> {
            Err(
                "the dependency-free CUDA driver loader is currently implemented for Windows"
                    .into(),
            )
        }

        pub fn memory_metrics(&self) -> CudaMemoryMetrics {
            CudaMemoryMetrics::default()
        }

        pub fn allocate(&self, _bytes: usize) -> Result<AcceleratorBuffer, String> {
            Err("CUDA allocation is unavailable on this platform build".into())
        }

        pub fn write_buffer(
            &self,
            _buffer: &AcceleratorBuffer,
            _offset: usize,
            _bytes: &[u8],
        ) -> Result<(), String> {
            Err("CUDA buffer writes are unavailable on this platform build".into())
        }

        #[allow(clippy::too_many_arguments)]
        pub fn standard_attention_bf16(
            &self,
            _keys: &AcceleratorBuffer,
            _values: &AcceleratorBuffer,
            _capacity: usize,
            _cache_len: usize,
            _query: &[f32],
            _heads: usize,
            _kv_heads: usize,
            _head_dim: usize,
            _value_dim: usize,
            _softmax_scale: f32,
            _window: Option<usize>,
        ) -> Result<Vec<f32>, String> {
            Err("CUDA BF16 standard attention is unavailable on this platform build".into())
        }

        pub fn int4_group32_matvec(
            &self,
            _packed: &[u32],
            _scales: &[f32],
            _rows: usize,
            _cols: usize,
            _input: &[f32],
        ) -> Result<Vec<f32>, String> {
            Err("CUDA INT4 kernel is unavailable on this platform build".into())
        }

        pub fn int4_group32_bf16_matvec(
            &self,
            _packed: &[u32],
            _scales: &[u16],
            _rows: usize,
            _cols: usize,
            _input: &[f32],
        ) -> Result<Vec<f32>, String> {
            Err("CUDA INT4/BF16 kernel is unavailable on this platform build".into())
        }

        pub fn int4_group32_bf16_bytes_matmul(
            &self,
            _packed: &[u8],
            _scales: &[u8],
            _rows: usize,
            _cols: usize,
            _positions: usize,
            _input: &[f32],
        ) -> Result<Vec<f32>, String> {
            Err("CUDA INT4/BF16 batched kernel is unavailable on this platform build".into())
        }

        pub fn resident_int4_group32_bf16_matvec(
            &self,
            _resident: &AcceleratorBuffer,
            _packed_offset: usize,
            _scale_offset: usize,
            _rows: usize,
            _cols: usize,
            _input: &[f32],
        ) -> Result<Vec<f32>, String> {
            Err("resident CUDA INT4/BF16 kernel is unavailable on this platform build".into())
        }

        pub fn resident_int4_group32_bf16_matmul(
            &self,
            _resident: &AcceleratorBuffer,
            _packed_offset: usize,
            _scale_offset: usize,
            _rows: usize,
            _cols: usize,
            _positions: usize,
            _input: &[f32],
        ) -> Result<Vec<f32>, String> {
            Err(
                "resident CUDA INT4/BF16 batched kernel is unavailable on this platform build"
                    .into(),
            )
        }

        pub fn bf16_matvec_bytes(
            &self,
            _weights_bf16: &[u8],
            _rows: usize,
            _cols: usize,
            _input: &[f32],
        ) -> Result<Vec<f32>, String> {
            Err("CUDA BF16 kernel is unavailable on this platform build".into())
        }

        pub fn bf16_matmul_bytes(
            &self,
            _weights_bf16: &[u8],
            _rows: usize,
            _cols: usize,
            _positions: usize,
            _input: &[f32],
        ) -> Result<Vec<f32>, String> {
            Err("CUDA BF16 batched kernel is unavailable on this platform build".into())
        }

        pub fn resident_bf16_matmul(
            &self,
            _resident: &AcceleratorBuffer,
            _weight_offset: usize,
            _rows: usize,
            _cols: usize,
            _positions: usize,
            _input: &[f32],
        ) -> Result<Vec<f32>, String> {
            Err("resident CUDA BF16 batched kernel is unavailable on this platform build".into())
        }

        pub fn bf16_transpose_matvec_bytes(
            &self,
            _weights_bf16: &[u8],
            _rows: usize,
            _cols: usize,
            _input: &[f32],
        ) -> Result<Vec<f32>, String> {
            Err("CUDA BF16 transpose kernel is unavailable on this platform build".into())
        }

        pub fn bf16_transpose_matmul_bytes(
            &self,
            _weights_bf16: &[u8],
            _rows: usize,
            _cols: usize,
            _positions: usize,
            _input: &[f32],
        ) -> Result<Vec<f32>, String> {
            Err("CUDA BF16 transpose batched kernel is unavailable on this platform build".into())
        }

        pub fn expert_swiglu_bf16(
            &self,
            _hidden: &[f32],
            _gate: PackedInt4Bf16View<'_>,
            _up: PackedInt4Bf16View<'_>,
            _down: PackedInt4Bf16View<'_>,
        ) -> Result<Vec<f32>, String> {
            Err("CUDA expert kernel is unavailable on this platform build".into())
        }
    }

    impl Accelerator for CudaAccelerator {
        fn name(&self) -> &str {
            "cuda-unavailable"
        }

        fn upload(&self, _key: ExpertKey, _bytes: &[u8]) -> Result<AcceleratorBuffer, String> {
            Err("CUDA driver backend is unavailable on this platform build".into())
        }

        fn release(&self, _buffer: &AcceleratorBuffer) {}
    }
}

pub use platform::CudaAccelerator;
