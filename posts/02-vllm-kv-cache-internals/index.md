---
title: "vLLM KV Cache Internals: The Engine, The Blocks, and What I Got Wrong the First Time"
date: 2026-04-10
draft: false
tags: ["vllm", "kv-cache", "inference", "paged-attention", "llm"]
categories: ["ml-engineering"]
description: "A corrected deep-dive into vLLM's KV cache manager — separating the scheduler from the allocator, fixing the prefix-caching story, and drawing everything out."
author: "Wojciech Wais"
cover:
  image: "inference_arch.png"
  alt: "LLM inference engine architecture"
---

## Why I came back to this

I wanted to revisit the KV cache manager because I realized my earlier framing was too convenient: I had effectively merged the scheduler, the cache manager, and the block allocator into one mental box. After reading the vLLM docs more carefully, it became clear there is a cleaner split — the scheduler owns admission and token-budget decisions, while the KV cache side exposes cache-specific operations such as looking up already-computed blocks and allocating or appending slots.

So this post is my second pass at the same topic. I am still using the "virtual memory for attention" analogy because it is genuinely useful, but I want to keep it as an analogy — not as a literal description of every internal detail.

---

## The engine boxes

![LLM inference engine high-level architecture](inference_arch.png)

The cleanest way I found to explain an inference engine is to draw three boxes, not two: `Scheduler`, `KVCacheManager`, and the GPU execution path that consumes block tables during attention. The scheduler decides which requests can make progress under current scheduling and cache constraints, while the cache layer answers cache-shaped questions such as "which prefix blocks are already computed?" and "can I allocate more slots for this request?"

That distinction matters because the scheduler is not just a thin wrapper around free-memory checks. The scheduler docs explicitly tie scheduling to budget limits and cache limits together, so reducing it to "it asks whether blocks are available" hides part of the real control logic. On the other side, the attention path does not care about policy at all — it consumes block-table mappings that let logical token positions resolve to physical KV storage inside the paged attention kernel.

---

## Why paging helps

![Memory waste: contiguous vs paged KV cache](memory_comparison.png)

The big problem PagedAttention solves is KV-cache waste from treating each request as if it needed one large contiguous reservation. vLLM's design stores a sequence's KV cache in non-contiguous blocks, which avoids the worst internal fragmentation from monolithic per-request allocations.

I still like the OS analogy: each request has a logical view of its KV blocks, and the engine maps those blocks to physical memory through block tables used by the attention kernel. But I do not want to over-claim that the whole engine behaves exactly like a general-purpose virtual memory subsystem — the serving stack still has batching, scheduling, and model-execution constraints layered on top.

One detail I would phrase more carefully than I did before: **block size is configurable**, not universally 16. The documented engine arguments support 8, 16, 32, 64, and 128 tokens per block. CUDA backends support up to 32 per block, while HPU defaults to 128. So when you read benchmarks quoting block-size-dependent numbers, it is worth checking what value was actually used.

---

## Logical blocks → physical memory

![Logical to physical KV block mapping via block table](kv_block_table.png)

Each request has a **logical block table** — an ordered list of KV blocks it owns, indexed from 0. At the physical level, those blocks can live anywhere in GPU VRAM, completely non-contiguously. The `block_table` tensor passed to the CUDA attention kernel is essentially a page table: it maps logical block index → physical block address, and the kernel uses it to look up the right memory offsets when computing attention.

The `block_table` itself is a plain `int32` tensor passed to the GPU. The kernel in `csrc/attention/attention_kernels.cu` resolves physical addresses as `block_table[seq_id][block_idx] * block_size + offset_in_block`. It is simple arithmetic, but that simplicity is load-bearing — it is what keeps the paged attention kernel fast enough not to be the throughput bottleneck.

---

## Inside the `BlockAllocator`

![BlockAllocator internal data structures and free pool](block_allocator.png)

There are two allocator variants in vLLM. The plain `UncachedBlockAllocator` manages a free list of physical blocks — allocate pops from the free pool, free pushes back. Simple.

The `CachedBlockAllocator` (used when prefix caching is on) is the more interesting one. Let's look at what it actually does and be precise about it, because this is where my first post was wrong.

---

## Prefix caching, corrected

This was the weakest part of my previous draft, so I rewrote it more conservatively.

vLLM's prefix caching design is built around hashing **full blocks only**. The hash for each block is computed from the block's tokens *and* the hash of its parent block — so it is a chain hash that encodes the full history from the beginning of the sequence, not just local content. This is stricter and more structured than a simple `{content_hash → PhysicalBlock}` lookup.

The other important correction is about mutation. The prefix-caching docs describe an append-only block table and mention that even temporarily duplicated full blocks can appear during allocation. This means "shared block, then copy-on-write when we append" is **not** a safe general explanation of how prefix reuse behaves in vLLM. The safer framing is:

- Full cached blocks can be reused across requests that share a common prefix.
- Appends happen by extending block-table state — not by casually editing shared historical blocks in place.
- `ref_count` tracks sharing, but the mutation model is more constrained than generic copy-on-write.

I still think "shared prefix reuse" is the right intuition for a broad audience, but I would not teach it as generic copy-on-write unless walking through the exact code path.

---

## How a request really moves

![Request lifecycle through KV cache manager](request_lifecycle.png)

The request lifecycle is still easiest to understand as prefill followed by decode, but I now want to phrase the mechanics less absolutely.

**Prefill** computes the full prompt in a single forward pass and populates KV state for all input tokens. It is compute-bound — one large matrix multiplication across the whole prompt.

**Decode** advances one token per step. It is memory-bandwidth-bound because each step loads KV state for the entire history. This is where throughput lives or dies.

So I would no longer write "decode allocates exactly one new block when the current block is full" as if that were the complete story. The more accurate framing:

- The block manager exposes slot-management and append operations.
- The **scheduler** decides whether to allow progress, subject to token budgets and cache constraints — not just raw free-block count.
- Preemption (swap to CPU or recompute from scratch) happens when those constraints cannot be satisfied, and the scheduler decides *which* strategy to use.

Having this in mind, the KV cache manager is less like a hidden memory trick and more like a **boundary layer between scheduling policy and physical KV storage**.

---

## What I want to check next

The thing I did not cover here is the swap-versus-recompute decision under memory pressure during long-context decoding. I have a feeling the cost crossover point depends heavily on PCIe generation and sequence length — and that is probably where the scheduler/cache boundary becomes most visible in real latency traces. I will try to profile this in the next post.
