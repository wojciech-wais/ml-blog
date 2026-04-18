---
title: "vLLM KV Cache Internals: The Engine, The Blocks, and What I Got Wrong the First Time"
date: 2026-04-10
draft: false
tags: ["vllm", "kv-cache", "inference", "paged-attention", "llm"]
categories: ["ml-engineering"]
description: "One another deep-dive into vLLM's KV cache manager — separating the scheduler from the allocator, fixing the prefix-caching story, and some diagrams."
author: "Wojciech Wais"
cover:
  image: "https://user-gen-media-assets.s3.amazonaws.com/seedream_images/ec254146-5ff3-4230-b50d-ecd78c296d2d.png"
  alt: "LLM inference engine high-level architecture"
  caption: "Excalidraw diagram: LLM inference engine — Scheduler, KVCacheManager, ModelRunner"
---

## Why I came back to this

I wanted to revisit the KV cache manager because I realized my earlier framing was too convenient: I had effectively merged the scheduler, the cache manager, and the block allocator into one mental box. After reading the vLLM docs more carefully, it became clear there is a cleaner split — the scheduler owns admission and token-budget decisions, while the KV cache side exposes cache-specific operations such as looking up already-computed blocks and calling `allocate_slots()` or `append_slots()`.

So this post is my second pass at the same topic. I am still using the "virtual memory for attention" analogy because it is genuinely useful, but I want to keep it as an analogy — not as a literal description of every internal.

---

## The engine boxes

![LLM inference engine high-level architecture](https://user-gen-media-assets.s3.amazonaws.com/seedream_images/ec254146-5ff3-4230-b50d-ecd78c296d2d.png)
*Three layers: the Scheduler decides policy, the KVCacheManager owns physical block bookkeeping, and the ModelRunner on the GPU just follows the block_table.*

The cleanest way I found to explain an inference engine is to draw three boxes, not two: `Scheduler`, `KVCacheManager`, and the GPU execution path that consumes block tables during attention. The scheduler decides which requests can make progress under current scheduling and cache constraints, while the cache layer answers cache-shaped questions such as "which prefix blocks are already computed?" and "can I allocate more slots for this request?"

That distinction matters because the scheduler is not just a thin wrapper around free-memory checks. The vLLM scheduler docs explicitly tie scheduling to **budget limits** and **cache limits** together, so reducing it to "it asks whether blocks are available" hides part of the real control logic. On the other side, the attention path does not care about policy at all — it consumes block-table mappings that let logical token positions resolve to physical KV storage inside the paged attention kernel.

---

## Why paging helps

![Memory waste: contiguous vs paged KV cache](https://user-gen-media-assets.s3.amazonaws.com/seedream_images/7665a0db-3435-468e-a6ec-b4c009b375e4.png)
*Left: the old world — contiguous per-request KV reservations waste 60–80% of VRAM. Right: paged blocks fill memory tightly with under 4% waste.*

The big problem PagedAttention solves is KV-cache waste from treating each request as if it needed one large contiguous reservation. vLLM's design stores a sequence's KV cache in non-contiguous blocks, which avoids the worst internal fragmentation from monolithic per-request allocations.

I still like the OS analogy: each request has a logical view of its KV blocks, and the engine maps those blocks to physical memory through block tables used by the attention kernel. But I do not want to over-claim that the whole engine behaves exactly like a general-purpose virtual memory subsystem — the serving stack still has batching, scheduling, and model-execution constraints layered on top.

One detail I would phrase more carefully than I did in my first post: **block size is configurable**, not universally 16. The vLLM engine arguments support 8, 16, 32, 64, and 128 tokens per block. CUDA backends support up to 32, while HPU defaults to 128. When you read benchmarks that are sensitive to block size, it is worth checking what value was actually used.

---

## Logical blocks → physical memory

![Logical to physical KV block mapping via block table](https://user-gen-media-assets.s3.amazonaws.com/seedream_images/380adff2-1b66-4930-bb64-386a3d4a11b4.png)
*Block table as page table: logical block index → physical block address. The CUDA kernel resolves addresses as `block_table[seq_id][block_idx] * block_size + offset_in_block`.*

Each request has a **logical block table** — an ordered list of KV blocks it owns, indexed from 0. At the physical level, those blocks can live anywhere in GPU VRAM, completely non-contiguously. The `block_table` tensor passed to the CUDA attention kernel is essentially a page table: logical block index → physical block address.

The `block_table` itself is a plain `int32` tensor passed to the GPU. The kernel in `csrc/attention/attention_kernels.cu` resolves physical addresses as:

```
phys_addr = block_table[seq_id][block_idx] * block_size + offset_in_block
```

It is simple arithmetic — but that simplicity is load-bearing. It is what keeps the paged attention kernel fast enough not to be the throughput bottleneck itself.

---

## Inside the `BlockAllocator`

![BlockAllocator internal data structures and free pool](https://user-gen-media-assets.s3.amazonaws.com/seedream_images/0639acdb-2574-4f37-8614-5c568bf86d40.png)
*Two allocator variants: `UncachedBlockAllocator` manages a plain free list; `CachedBlockAllocator` adds a chain-hash map for prefix reuse.*

There are two allocator variants in vLLM. The plain `UncachedBlockAllocator` manages a free list of physical blocks — `allocate()` pops from the pool, `free()` pushes back. Simple.

The `CachedBlockAllocator` (used when prefix caching is enabled) is the more interesting one. Let's be precise about what it does — because this is exactly where my first post was wrong.

---

## Prefix caching, corrected

This was the weakest part of my previous draft.

vLLM's prefix caching design is built around hashing **full blocks only**. The hash for each block is computed from the block's tokens *and* the hash of its parent block in the chain — so it encodes the full token history from the start of the sequence, not just local content. This is stricter and more structured than a simple `{content_hash → PhysicalBlock}` lookup.

The second correction is about mutation. The prefix-caching docs describe an append-only block table and mention that temporarily duplicated full blocks can appear during allocation. "Shared block → copy-on-write when we append" is **not** a safe general explanation of how prefix reuse works in vLLM. The safer framing:

- Full cached blocks can be **reused** across requests that share a common prefix.
- Appends happen by **extending block-table state** — not by mutating shared historical blocks.
- `ref_count` tracks sharing, but the mutation model is more constrained than generic OS copy-on-write.

I still think "shared prefix reuse" is the right intuition for a broad audience. I would just not teach it as generic copy-on-write unless walking through the exact code path in `CachedBlockAllocator`.

---

## How a request really moves

![Request lifecycle through KV cache manager](https://user-gen-media-assets.s3.amazonaws.com/seedream_images/23e78686-8d70-4d38-bdf2-8321306fda23.png)
*Prefill fills KV cache for the whole prompt in one pass; decode extends it one token at a time. Preemption (swap or recompute) kicks in when scheduler budgets are exhausted.*

The lifecycle is still prefill → decode, but I now want to phrase the mechanics less absolutely.

**Prefill** computes the full prompt in a single `forward()` call and populates KV state for all input tokens. It is compute-bound — one large matrix multiplication across the whole prompt.

**Decode** advances one token per step. It is memory-bandwidth-bound because each step loads KV state for the entire history. This is where throughput lives or dies.

I would no longer write "decode allocates exactly one new block when the current block is full" as if that were the full story. The more accurate framing:

- The block manager exposes slot-management and `append_slots()` operations.
- The **scheduler** decides whether to allow progress, subject to token budgets and cache constraints — not just raw free-block count.
- Preemption (swap blocks to CPU via `swap_out()`, or recompute from scratch) happens when those constraints cannot be satisfied. The scheduler decides which strategy to use.

Having this in mind, the KV cache manager is best understood as a **boundary layer between scheduling policy and physical KV storage** — not a memory trick bolted onto the side of the engine.

---

## What I want to check next

The thing I did not cover here is the swap-versus-recompute decision under real memory pressure during long-context decoding. I have a feeling the cost crossover point depends heavily on PCIe generation and sequence length — and that is probably where the scheduler/cache boundary becomes most visible in actual latency traces. I will try to profile this with Nsight Systems in the next post and see if the numbers match the intuition.
