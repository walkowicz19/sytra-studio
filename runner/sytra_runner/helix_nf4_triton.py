"""Optional CUDA decoding and fused frozen NF4 matrix multiplication."""
import torch
import triton
import triton.language as tl


@triton.jit
def _decode_kernel(packed, scales, codebook, output, start: tl.constexpr,
                   count: tl.constexpr, block_size: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < count
    position = start + offset
    byte = tl.load(packed + position // 2, mask=valid, other=0).to(tl.int32)
    code = tl.where(position % 2 == 0, byte & 15, byte >> 4)
    value = tl.load(codebook + code)
    scale = tl.load(scales + position // block_size, mask=valid, other=0)
    tl.store(output + offset, value * scale, mask=valid)


def decode_nf4(packed, scales, start, end, block_size, codebook, dtype):
    """Decode a contiguous element range to a newly allocated CUDA tensor."""
    if packed.device.type != 'cuda':
        raise ValueError('Triton NF4 decoder requires CUDA')
    count = end - start
    output = torch.empty(count, device=packed.device, dtype=dtype)
    if count:
        _decode_kernel[(triton.cdiv(count, 1024),)](packed, scales, codebook, output,
                                                    start, count, block_size, 1024)
    return output


@triton.jit
def _matmul_kernel(inputs, packed, scales, codebook, output,
                   M: tl.constexpr, IN: tl.constexpr, OUT: tl.constexpr,
                   BLOCK_SIZE: tl.constexpr, BACKWARD: tl.constexpr,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    reduction = tl.arange(0, BK)
    width: tl.constexpr = IN if BACKWARD else OUT
    depth: tl.constexpr = OUT if BACKWARD else IN
    accumulator = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(depth, BK)):
        k = start * BK + reduction
        a = tl.load(inputs + rows[:, None] * depth + k[None, :],
                    mask=(rows[:, None] < M) & (k[None, :] < depth), other=0)
        if BACKWARD:
            index = k[:, None].to(tl.int64) * IN + cols[None, :].to(tl.int64)
        else:
            index = cols[None, :].to(tl.int64) * IN + k[:, None].to(tl.int64)
        valid = (k[:, None] < depth) & (cols[None, :] < width)
        byte = tl.load(packed + index // 2, mask=valid, other=0).to(tl.int32)
        code = tl.where(index % 2 == 0, byte & 15, byte >> 4)
        value = tl.load(codebook + code)
        scale = tl.load(scales + index // BLOCK_SIZE, mask=valid, other=0)
        weight = (value * scale).to(a.dtype)
        accumulator += tl.dot(a, weight, input_precision='ieee')
    tl.store(output + rows[:, None] * width + cols[None, :], accumulator,
             mask=(rows[:, None] < M) & (cols[None, :] < width))


def matmul_nf4(inputs, packed, scales, in_features, out_features, block_size, codebook, *, backward=False):
    """Multiply by frozen NF4 weights, decoding each tile inside the kernel."""
    inputs = inputs.contiguous()
    rows = inputs.shape[0]
    width = in_features if backward else out_features
    output = inputs.new_empty((rows, width))
    if rows:
        _matmul_kernel[(triton.cdiv(rows, 16), triton.cdiv(width, 32))](
            inputs, packed, scales, codebook, output, rows, in_features, out_features,
            block_size, backward, 16, 32, 32, num_warps=4)
    return output
