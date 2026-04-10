---
title: "Digging Into vLLM's KV Cache Manager"
date: 2026-04-01
draft: false
tags: ["vllm", "kv-cache", "inference", "paged-attention", "llm", "memory"]
categories: ["ml-engineering"]
description: "I noticed throughput plateauing under mixed-length traffic. So I sat down and actually read the vLLM source. This is what I found."
author: "Wojciech Wais"
ShowToc: true
TocOpen: false
---

## Why I Went Looking

I've been running Mistral-7B under mixed-length traffic for a few weeks now and noticed throughput plateauing earlier than I'd expect from GPU utilization alone. My first instinct was memory fragmentation in the KV cache layer, so I sat down to actually read the vLLM source and docs rather than just tweak flags. What follows is what I found, organized from first principles outward — because trying to understand the block manager without first understanding what it manages turned out to be a trap I fell into on my first pass.

---

## Layer 1: What KV Cache Actually Does

Start with autoregressive decoding. To generate token N, the transformer's attention layers need to look at all N-1 previous tokens. That means attending over K and V tensors for the full context. Without any caching, every decode step would recompute K and V from scratch for every past token — making decode quadratic in sequence length.

KV caching solves this by storing the K and V tensors from each attention layer as they're produced, then reusing them at every subsequent decode step. The computation at step N becomes: compute K/V only for the *new* token, concatenate it with the cached K/V from steps 1..N-1, run attention. So the decode-step cost is proportional to the *context length* (we still attend over the full cached prefix), but we avoid *recomputing* those K/V tensors from token embeddings every time.

This distinction matters, and I want to be precise about it: **KV caching eliminates redundant K/V computation, but each decode step still attends over the entire cached context.** Decode cost is O(context length) per step, not O(1). That's an important thing to get right before reading any further.

The cost of this caching is memory. For LLaMA-2-13B in FP16 — 40 layers, 40 attention heads, `d_head=128` — the per-token KV footprint is roughly **0.78 MiB**. A single 4096-token sequence costs ~3.1 GiB. On a 40 GB A100 with ~26 GB for model weights, you have perhaps 12 GB left for KV cache. That limits you to a handful of concurrent sequences. Memory capacity, not compute, is the concurrency ceiling — and that's the root problem the KV cache manager exists to solve.

---

## Layer 2: PagedAttention and the Block Allocator

The naive solution to the memory problem is to preallocate a contiguous KV buffer for each incoming request, sized to the maximum sequence length. Simple, predictable — and brutally wasteful. A request that finishes in 50 tokens but was given a 2048-token slab wastes 97.5% of its allocation for its entire lifetime. After many requests of varying lengths complete and return memory, the free space becomes a patchwork of non-contiguous holes. A new request might fail to allocate even when there's technically enough free bytes total. This is textbook external fragmentation.

vLLM solves this with **PagedAttention**, introduced in [Kwon et al., 2023](https://arxiv.org/abs/2309.06180). The idea is borrowed directly from OS virtual memory paging: instead of one contiguous slab per request, divide all KV cache memory into fixed-size **physical blocks** and allocate them on demand from a shared global pool. Each request carries a **block table** — a logical-to-physical block index mapping. The attention kernel follows this table to reassemble the K/V tensors at runtime, even when the underlying physical blocks are scattered across GPU memory.

The OS analogy maps cleanly here, with one important clarification:

| OS Virtual Memory | vLLM PagedAttention |
|---|---|
| Virtual page | Logical KV block (per request) |
| Physical frame | Physical KV block (by block ID) |
| Page table | Per-request block table |
| Frame allocator | `BlockPool` free queue |
| LRU eviction | LRU block eviction + request preemption |

Worth noting: this is not a true virtual memory system with hardware faulting. Block allocation happens eagerly, coordinated by the scheduler *before* the request runs. The analogy is about the data structure pattern, not the runtime fault model.

Internal fragmentation is reduced to at most `B-1` wasted token slots per request (where B is block size), and external fragmentation disappears entirely since all blocks are identical in size.

---

## Layer 3: Automatic Prefix Caching — A Separate Feature

This is where I muddled things in my first draft. Prefix caching (APC) is **not** the same mechanism as basic KV caching. It's an opt-in feature, enabled with `--enable-prefix-caching` or `enable_prefix_caching=True` in the engine config. It builds on top of the block allocator to enable *cross-request* reuse of K/V blocks that were computed for a shared prompt prefix.

The mechanism: when a request finishes, its blocks are returned to the free pool but not immediately cleared. They're indexed by a content hash. When a new request arrives with a prompt that starts with the same tokens, the scheduler calls `get_computed_blocks()`, which walks the hash chain and identifies which prefix blocks already exist in GPU memory. Those blocks are marked as in-use again without recomputing them — effectively skipping their portion of prefill entirely.

The block hash is defined as `hash(prefix_tokens + block_tokens)` with chaining: block N's hash incorporates the hash of block N-1. This ensures two sequences only share a cached block at position N if they've had *identical* token content at every position 0..N-1. No false hits. The hash also incorporates extra keys — LoRA adapter IDs, multimodal content hashes — so blocks computed under adapter A are never reused for adapter B requests.

To be clear about what APC does and doesn't do: it saves **prefill computation** for the shared prefix. It does **not** make decode cheaper. Decode still attends over the full cached context, APC'd or not.

---

## Internals of the Block Manager

> Note: based on the vLLM v1 source tree as of early 2026. Internal APIs move fast — treat concepts as stable but verify class names on your version.

### `KVCacheBlock`

The atomic unit. Each instance tracks:
- `block_id` — index into the preallocated GPU tensor
- `ref_cnt` — when this hits 0, the block is an eviction candidate
- `block_hash` — optional, set only when APC is enabled

### `FreeKVCacheBlockQueue`

A custom doubly-linked list rather than a Python `deque`. The reason is O(1) removal from *any* position: when APC gets a cache hit and needs to pull a specific block out of the eviction pool, a deque requires O(n) traversal. It's a small thing but this is a hot path.

### `BlockPool`

The allocation interface. Holds the full set of `KVCacheBlock` objects, the free queue, and a hash-to-block map for APC lookups. When `allocate()` is called, it pops from the front of the free queue. If APC is on and the popped block has a cached hash, the block is evicted lazily — hash table entry removed, metadata cleared — before being handed out.

### `KVCacheManager` and `allocate_slots()`

The scheduler-facing interface. `allocate_slots()` checks for prefix hits, calculates how many new blocks are needed, verifies free pool capacity, and either commits the allocation or returns `None` to signal preemption. Preemption policy lives in the scheduler; the cache manager is only the mechanism. That separation is one of the cleaner design decisions in the codebase.

---

## Back to My Original Problem

After actually reading this properly, I think the throughput plateau was likely low APC hit rate — the system was evicting prefix blocks before they could be reused, because arriving requests had variable enough prefixes that nothing was staying resident. The most defensible lever is maximizing shared prefix length at the application level (consistent system prompts, consistent prompt templates) and tuning `gpu_memory_utilization` to give the pool more headroom.

I haven't run a controlled ablation on this yet. The honest position is: I have a hypothesis, not a measurement. I'll write a follow-up once I have actual traces.
