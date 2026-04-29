---
title: "Chasing Memory Bandwidth: Fused CUDA Kernels for LLM Inference"
date: 2026-04-29
draft: false
tags: ["cuda", "kernels", "inference", "memory-bandwidth", "triton", "llm", "optimization", "pytorch"]
categories: ["ml-engineering"]
description: "Profiling a LLaMA-3 forward pass revealed 90% of the GPU sitting idle. Here is how I chased that down with fused CUDA kernels — RMSNorm, RoPE, SwiGLU, and INT8 KV-cache quantization — reaching 85–95% memory bandwidth utilization."
author: "Wojciech Wais"
ShowToc: true
TocOpen: false
---

I have been staring at profiler traces for the last few weeks trying to figure out why our inference stack is so slow. Not slow in a "model architecture needs redesign" way — slow in a "we're leaving 90% of the GPU on the table" way. So I decided to dig properly into the memory access patterns of the core transformer ops and see how far I can push things with custom kernels.

The setup: NVIDIA A100 40GB, PyTorch 2.3, CUDA 12.1, a small repo I called `fused-kernels-llm`. Let me walk through what I found.

## The thing that started it all

I was profiling a standard LLaMA-3 forward pass with PyTorch Profiler and something immediately caught my eye. The RMSNorm + residual add at every transformer layer was showing up as a cluster of tiny, back-to-back kernel launches, each barely a few microseconds long. The compute utilization? Embarrassing. Under 10%.

So the question is — where does the time actually go? Not in computation. In memory. The A100 can do 312 TFLOP/s in FP32, but its memory bandwidth is capped at 1.55 TB/s. For something like RMSNorm, the arithmetic intensity is painfully low: you read the input vector, compute a norm, scale it, write it back. Maybe 3 memory operations per float op. That puts us deep in memory-bound territory on the roofline model — and standard PyTorch makes it worse by splitting the operation into four separate kernel launches:

```python
hidden_states = input + residual                              # round-trip 1
variance = hidden_states.pow(2).mean(-1, keepdim=True)        # round-trip 2
hidden_states = hidden_states * torch.rsqrt(variance + eps)   # round-trip 3
output = hidden_states * weight                               # round-trip 4
```

Four passes over the same data. The GPU's DRAM is getting hammered on every single one of these. Having this in mind, the fix feels obvious in principle — fuse all of this into a single kernel. But the devil, as usual, is in the implementation.

## First attempt: naive CUDA kernel

I started with the simplest thing that could possibly work. A naive `__global__` kernel where each CUDA thread handles one row of the hidden state tensor, doing the whole residual + norm in one shot:

```cpp
__global__ void naive_rmsnorm_residual(
    const float* input, const float* residual,
    const float* weight, float* output,
    float eps, int hidden_size
) {
    int batch = blockIdx.x;
    int base = batch * hidden_size;

    float sum = 0.0f;
    for (int i = 0; i < hidden_size; i++) {
        float val = input[base + i] + residual[base + i];
        sum += val * val;
    }
    float inv_rms = rsqrtf(sum / hidden_size + eps);
    for (int i = 0; i < hidden_size; i++) {
        float val = input[base + i] + residual[base + i];
        output[base + i] = val * inv_rms * weight[i];
    }
}
```

Correctness: good. Performance: not great. I'm reading `input` and `residual` twice — once to accumulate the variance, once to normalize. And I'm doing it with scalar loads, which leaves vectorization bandwidth on the table.

Let's check what the profiler says. With Nsight Systems I could immediately see the memory throughput sitting around 30–40% of theoretical peak. Not terrible for a naive kernel but clearly fixable.

## Vectorized loads and the `float4` trick

The first real optimization I tried was switching to vectorized `float4` loads. Modern GPUs can load 128 bits in a single transaction — that's 4 floats at once — so doing scalar `float` loads wastes 75% of available bandwidth per transaction:

```cpp
float4 vec_in  = reinterpret_cast<const float4*>(input)[idx / 4];
float4 vec_res = reinterpret_cast<const float4*>(residual)[idx / 4];
float a = vec_in.x + vec_res.x;
float b = vec_in.y + vec_res.y;
float c = vec_in.z + vec_res.z;
float d = vec_in.w + vec_res.w;
```

Worth to note: this only works cleanly when your hidden dimension is divisible by 4, which it is for every model I care about (4096 for LLaMA-3, 3072 for Mistral-7B, etc.). I still added a scalar fallback path for boundary elements to be safe.

After this change, bandwidth utilization jumped to around 65%. Better, but the roofline says we can do more.

## The double-pass problem: Welford's algorithm

The naive kernel reads the input twice — once for variance, once for normalization. I thought maybe I could fuse this into a single pass using Welford's online variance algorithm, which accumulates mean and M2 in one sweep:

```cpp
float mean = 0.0f, m2 = 0.0f;
for (int i = tid; i < hidden_size; i += blockDim.x) {
    float val = input[base + i] + residual[base + i];
    float delta = val - mean;
    mean += delta / (i / blockDim.x + 1);
    float delta2 = val - mean;
    m2 += delta * delta2;
}
float variance = m2 / hidden_size;
```

The catch: now I need a parallel reduction across threads in the block before I can do the normalization pass. This is where warp shuffle instructions come in. Instead of round-tripping through shared memory, I can use `__shfl_xor_sync` to reduce within a warp:

```cpp
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}
```

So the actual memory traffic per element goes from two full reads to one. I was honestly a bit worried this would hurt numerical stability on BF16, but after running the test suite against the PyTorch reference at `rtol=1e-3`, it held up fine.

## Bandwidth numbers after fusion

After combining vectorized loads, Welford's single-pass reduction, and the warp shuffle reduction, I got to the numbers I was hoping for:

| Operation | Input Shape | Speedup vs PyTorch | Bandwidth Efficiency |
|---|---|---|---|
| RMSNorm + Residual | [32, 2048, 4096] | **3.4×** | 92% |
| RoPE | [32, 16, 2048, 128] | **2.7×** | 88% |
| SwiGLU | [32, 2048, 11008] | **2.2×** | 89% |
| KV-cache INT8 quant | [32, 2048, 4096] | — | 95% |

Accuracy vs FP32 PyTorch reference is 1e-6 relative tolerance for the floating-point ops, and less than 2% accuracy degradation from INT8 quantization.

## RoPE: the access pattern mess

RoPE was trickier than I expected. The rotation mixes pairs of elements across the head dimension:

```cpp
float cos_val = cos_cache[pos * head_dim / 2 + i];
float sin_val = sin_cache[pos * head_dim / 2 + i];
float q_new_r = q_real * cos_val - q_imag * sin_val;
float q_new_i = q_imag * cos_val + q_real * sin_val;
```

The problem is that `q_real` and `q_imag` sit at non-adjacent memory offsets by default (the first and second halves of the head), which causes cache misses if you're not careful about how you index. I spent an embarrassing amount of time here because the naive index calculation I wrote first was hitting ~40% bandwidth — worse than unfused PyTorch. The fix was interleaving the real/imaginary layout in the kernel rather than assuming the half-split layout. Once I sorted that, it jumped to 88%.

## SwiGLU: surprisingly straightforward

Compared to RoPE, SwiGLU was almost anticlimactic. The formula is `silu(gate) * up` where `silu(x) = x * sigmoid(x)`. With two input tensors and one output, this fuses cleanly:

```cpp
float gate_val  = gate[idx];
float up_val    = up[idx];
float silu_gate = gate_val * __fdividef(1.0f, 1.0f + __expf(-gate_val));
output[idx]     = silu_gate * up_val;
```

The `__expf` fast-math approximation introduces negligible error for this use case. Worth to note: I tested with `__expf` vs `expf` and saw a ~6% throughput difference on A100, so it's worth using where precision allows it.

## KV-cache quantization: where atomics get hairy

The INT8 KV-cache quantization was the last piece, and probably the most annoying to get right. The idea is to reduce KV-cache memory by ~4× by quantizing to INT8 with per-channel scaling factors. The tricky part is computing the per-channel absmax in parallel:

```cpp
__global__ void compute_scales(float* input, float* scales, int channels) {
    int channel = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x + channel * n_per_channel;
    if (idx < n_per_channel * (channel + 1))
        atomicMax((int*)&scales[channel], __float_as_int(fabsf(input[idx])));
}
```

I was stuck on this for a day because `atomicMax` doesn't support `float` directly — you have to use the integer alias trick with `__float_as_int`, which only works for non-negative floats (which `fabsf` guarantees). Not sure if this is something that will bite me on edge cases I haven't thought of yet, but the test suite is passing. The 4.2× memory reduction is real and measurable with `nvtop` during a forward pass with sequence length 2048.

## Triton as a fallback

I wasn't confident about maintaining hand-written CUDA kernels across every GPU architecture we might hit in CI, so I also wrote Triton versions as a fallback. Triton's autotuning handles block size selection across architectures automatically:

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': bs}, num_warps=nw, num_stages=ns)
        for bs in [64, 128, 256, 512]
        for nw in [2, 4, 8]
        for ns in [2, 3, 4]
    ],
    key=['n_elements']
)
@triton.jit
def fused_rmsnorm_residual_kernel(
    input_ptr, residual_ptr, weight_ptr, output_ptr,
    n_elements, eps, BLOCK_SIZE: tl.constexpr
):
    pid     = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < n_elements
    x       = tl.load(input_ptr + offsets, mask=mask)
    r       = tl.load(residual_ptr + offsets, mask=mask)
    hidden  = x + r
    variance = tl.sum(hidden * hidden) / n_elements
    # ... normalization and weight application
```

In practice, the Triton versions land at around 80–88% of the hand-tuned CUDA performance — good enough for most deployment targets. The fallback hierarchy in Python looks like this:

```python
def fused_rmsnorm_residual(input_tensor, residual, weight, eps=1e-6):
    try:
        return _C.fused_rmsnorm_residual_cuda_optimized(input_tensor, residual, weight, eps)
    except (RuntimeError, ImportError):
        try:
            return fused_rmsnorm_residual_triton(input_tensor, residual, weight, eps)
        except Exception:
            return _pytorch_reference_impl(input_tensor, residual, weight, eps)
```

## `torch.compile` compatibility

One thing I did not anticipate being painful: making custom ops work with `torch.compile()`. You have to register them explicitly through `torch.library.custom_op` or the graph compiler will either refuse them or silently fall back to eager:

```python
@torch.library.custom_op("fused_kernels::rmsnorm_residual", mutates_args=())
def rmsnorm_residual_op(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    return fused_rmsnorm_residual(input, residual, weight, eps)
```

Once registered properly, `torch.compile` treats it as an opaque node and doesn't try to decompose it.

## Multi-precision support

I had to template the CUDA kernels to handle FP32, FP16, and BF16 separately. The accumulation logic is different for each — you want to accumulate in FP32 even when loading BF16 to avoid precision loss during the variance reduction:

```cpp
template<typename scalar_t>
__global__ void fused_rmsnorm_residual_kernel(
    const scalar_t* input,
    const scalar_t* residual,
    const scalar_t* weight,
    scalar_t* output,
    float eps,
    int hidden_size
);

template __global__ void fused_rmsnorm_residual_kernel<float>(...);
template __global__ void fused_rmsnorm_residual_kernel<c10::Half>(...);
template __global__ void fused_rmsnorm_residual_kernel<c10::BFloat16>(...);
```

So the final bandwidth story is: 85–95% across all four kernel families, measured against the A100's theoretical 1555 GB/s peak. The roofline plots confirm that every op sits right against the memory-bound ceiling, which is about as good as you can get without restructuring the algorithms themselves.

The thing I'm still not fully satisfied with is the multi-layer fusion story. Right now I'm fusing within individual operations, but fusing an entire transformer sublayer — attention + MLP + two norms in one kernel — would theoretically reduce memory traffic even further. I have a feeling the scheduling complexity would be non-trivial though. I will try to cover that in the next post.
