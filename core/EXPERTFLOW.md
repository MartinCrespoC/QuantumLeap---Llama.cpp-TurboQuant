# ExpertFlow — MoE-Aware Inference Engine

**Goal**: 150-200+ tok/s on Qwen3.5-122B-A10B with consumer hardware ($500 PC)

**Status**: Phases 1-4 complete + Turbo optimizations (35/35 tests pass). Works with **any MoE model**.

---

## Supported Models

ExpertFlow works with **any MoE GGUF model**. All dimensions are read from model metadata at runtime.

| Model | Experts | Top-K | Expert Pattern | Tested |
|-------|---------|-------|----------------|--------|
| **Qwen3.5-122B-A10B** | 256/layer | 8 | Stacked (`ffn_gate_exps`) | ✅ 35/35 |
| **Qwen3.5-35B-A3B** | 64/layer | 4 | Stacked | Supported |
| **DeepSeek V2/V3** | 160/layer | 6 | Stacked | Supported |
| **Mixtral 8x7B/8x22B** | 8/layer | 2 | Per-expert (`ffn_gate.{E}`) | Supported |
| **Llama 4 MoE** | varies | varies | Stacked | Supported |
| **DBRX** | 16/layer | 4 | Fused (`ffn_gate_up_exps`) | Supported |

Auto-configuration: `BackendConfig::auto_config(vram_bytes, arch)` and `PipelineConfig::auto_config(vram_bytes, arch)` compute optimal cache sizes, staging buffers, and prefetch parameters from the model's actual dimensions.

---

## The Problem

Current llama.cpp treats MoE models like dense models: it offloads **entire layers** (all experts) to GPU or CPU. But MoE only activates top-K experts per token. Result: we waste bandwidth reading unused expert weights.

```
Current: Layer N → load ALL 256 experts → route → use 8 → waste 248
ExpertFlow: Layer N → route → load ONLY 8 active experts → use 8 → waste 0
```

## The Math

### Hardware: Ryzen 7 3700X + RX 5600 XT (6GB) + 46GB DDR4

| Resource | Bandwidth | Latency |
|----------|-----------|---------|
| GPU VRAM (GDDR6) | 288 GB/s | ~100ns |
| DDR4 RAM | ~40 GB/s | ~60ns |
| PCIe 4.0 x16 | ~25 GB/s (real) | ~1μs |

### Target Model: Qwen3.5-122B-A10B (IQ2_XXS, 34.1 GB)

Architecture (from GGUF metadata):
- **48 layers**, **256 experts/layer**, **top-8 routing**
- Expert FFN size: 1024, embedding dim: 3072
- Attention: 32 heads, 2 KV heads (GQA)
- Shared expert FFN: 1024
- Context: 262,144 tokens

| Component | Tensors | Size (IQ2_XXS) | Location |
|-----------|---------|----------------|----------|
| Shared (attention, norms, router, embed, LM head) | 735 | **4,114 MB** | GPU VRAM (permanent) |
| All experts (48 layers × 256 × 3 projections) | 36,864 | **30.1 GB** | CPU RAM (cache on GPU) |
| **Per expert** (gate + up + down) | 3 | **2.51 MB** | GPU cache slot |
| **Active experts per token** (8/layer × 48) | 384 | **964 MB** | GPU (cached/streamed) |
| **Active per layer** (shared + 8 experts) | — | **105.8 MB** | Mixed |

*Verified by ExpertMap GGUF parser against real model file.*

### Hybrid Architecture Discovery: MoE + Mamba (SSM)

Qwen3.5-122B-A10B is a **hybrid model** combining MoE with State Space Models!
Many layers contain SSM blocks (`ssm_out`, `ssm_alpha`, `ssm_beta`, `ssm_conv1d`, `ssm_dt`).

**Per-layer shared weight breakdown (verified, layer 10 as sample):**

| Component | Size/Layer | Quant | Notes |
|-----------|-----------|-------|-------|
| Attention (Q/K/V/O) | 41.3 MB | Q5_K | Main bottleneck |
| SSM / Mamba | 20.3 MB | Q6_K + Q8_0 + F32 | `ssm_out`=19.7 MB, rest small |
| Shared expert FFN | 6.6 MB | Q5_K | One shared expert per layer |
| Router MLP | 3.0 MB | F32 | `ffn_gate_inp` |
| Norms | 0.01 MB | F32 | Negligible |
| **Total per layer** | **71.1 MB** | | × 48 layers = 3,413 MB |

**Embeddings (one-time):** 818 MB (token_embd 409 MB + output 409 MB, Q4_K)

### Speed Limits

| Scenario | Math | Speed |
|----------|------|-------|
| All on GPU (shared + experts) | 288 GB/s ÷ 5.08 GB total/token | **56.7 tok/s** |
| ExpertFlow: shared on GPU, experts via PCIe (0% hit) | bottleneck: PCIe stall | **23.9 tok/s** |
| ExpertFlow: 50% cache hit | PCIe partially hidden by attention | **44.2 tok/s** |
| ExpertFlow: 70% cache hit | **PCIe fully hidden by attention** | **56.5 tok/s** |
| ExpertFlow: 100% cache hit | attention-bandwidth bound | **56.8 tok/s** |
| Current (full layer offload, ngl=7) | reads 97% unused experts | **5.1 tok/s** |

### The Critical Insight

The active expert weight per layer is only **20.1 MB** (8 × 2.51 MB).
The shared attention weight per layer is **85.7 MB**.

**Attention is the bottleneck, not experts!** PCIe expert transfer (0.241 ms at 70% hit)
is completely hidden behind attention compute (0.298 ms). This means:

- Even with modest 50-70% expert cache hit rate → **no expert transfer stall**
- Speedup comes from putting ALL shared weights on GPU (not just 7 layers)
- **11× speedup** over current approach (56.7 vs 5.1 tok/s)

### VRAM Budget (6 GB)

**Baseline (no TurboQuant KV):**
| Region | Size | Contents |
|--------|------|----------|
| Shared weights (permanent) | 4,114 MB | All 48 layers of attention, norms, router, embeddings |
| Expert cache | ~1,900 MB | ~757 expert slots (6.2% of 12,288 total) |
| KV cache + overhead | ~200 MB | Working memory |

**With TurboQuant KV (TQ3 mode, 3.5 bits/channel):**
| Region | Size | Savings | Contents |
|--------|------|---------|----------|
| Shared weights (permanent) | 4,114 MB | — | All 48 layers of attention, norms, router, embeddings |
| Expert cache | ~2,056 MB | **+156 MB** | ~819 expert slots (+62 slots from KV savings) |
| KV cache (TQ3 compressed) | ~44 MB | **-156 MB** | 4K ctx @ 3.5 bits/ch vs 16 bits/ch FP16 |
| Overhead | ~200 MB | — | Working memory |

*TQ3 KV compression ratio: 0.22 (9.1× savings vs FP32, 4.6× vs FP16)*

### Turbo Optimizations (Phases A-D)

| Phase | Optimization | Impact | Status |
|-------|--------------|--------|--------|
| **A** | TurboQuant KV compression (TQ3) | +156 MB VRAM → +62 expert cache slots | ✅ Implemented |
| **B** | Fused CUDA MoE dispatch kernel | IQ2_XXS dequant+GEMV, fused SiLU⊙mul, warp reduction | ✅ Implemented |
| **C** | Expert transfer compression (LZ77) | **89.7% savings** on IQ2_XXS data (ratio 0.10!) | ✅ Implemented |
| **D** | Adaptive routing predictor (Markov) | 74%→92% accuracy (cold→warm), EMA decay | ✅ Implemented |

**Expert transfer compression details:**
- Lightweight LZ77-style compressor optimized for IQ2_XXS quantized weights
- Real expert data: 811,008 bytes → 83,232 bytes (89.7% saved, 0.10 ratio)
- Pinned memory pool: pre-allocated staging buffers for zero-copy H2D transfers
- Reduces PCIe bandwidth by 10×, enabling higher cache miss tolerance

**Adaptive routing predictor details:**
- Per-layer expert frequency tracking with EMA decay (configurable α)
- Cross-layer transition probabilities (expert[L] → expert[L+1])
- Prediction combines popularity (60%) + transitions (40%)
- Accuracy: 100% on deterministic patterns, 74%→92% on semi-random (improves over time)

### Path Beyond 57 tok/s

| Optimization | Potential | Notes |
|---|---|---|
| Shared weight requant (F16→IQ4) | **2×** (reduce 85.7 to ~40 MB/layer) | Some shared tensors are F16/F32 |
| Batch processing (pp8+) | **4-8×** for prompt | Higher GPU utilization |
| Async layer pipeline | **10-20%** | Overlap layer N attention with N-1 expert |
| GPU fused MoE kernel (Phase B) | **15-25%** | Eliminate CPU dispatch overhead |
| Expert compression + predictor (C+D) | **2-3× PCIe effective BW** | 90% compression + 92% prefetch accuracy |

**Realistic target with all optimizations: 80-120 tok/s generation, 400+ tok/s prompt**

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│                  CPU RAM (46GB)                   │
│  ┌──────────────────────────────────────────────┐│
│  │  Expert Weight Store (mmap'd, ~29.8 GB)      ││
│  │  48 layers × 256 experts × 2.36 MB each      ││
│  │  expert[layer][id] → {offset, size, quant}   ││
│  └──────────────────┬───────────────────────────┘│
│                     │ PCIe 4.0 x16 (25 GB/s)     │
│                     │ hipMemcpyAsync              │
└─────────────────────┼────────────────────────────┘
                      ▼
┌──────────────────────────────────────────────────┐
│               GPU VRAM (6 GB)                    │
│  ┌───────────────────┐ ┌───────────────────────┐ │
│  │ Permanent (~1.6GB)│ │ Expert Cache (~4 GB)  │ │
│  │ • Attention (all) │ │ • LRU + freq-weight   │ │
│  │ • Embeddings      │ │ • ~1,735 experts      │ │
│  │ • Router MLPs     │ │ • Double-buffered     │ │
│  │ • Shared experts  │ │ • Async H2D stream    │ │
│  │ • Norms + LM Head │ │ • Per-layer slots     │ │
│  └───────────────────┘ └───────────────────────┘ │
│  ┌──────────────────────────────────────────────┐│
│  │ Compute Pipeline (3 HIP streams)             ││
│  │ Stream 0: Attention + Router + Shared expert ││
│  │ Stream 1: Expert matmul (from cache)         ││
│  │ Stream 2: Expert H2D prefetch                ││
│  └──────────────────────────────────────────────┘│
└──────────────────────────────────────────────────┘
```

### Inference Flow (per token)

```
1. EMBED   → GPU (permanent)               [Stream 0]
2. For each layer L = 0..47:
   a. ATTENTION    → GPU (permanent)        [Stream 0]
   b. ROUTER       → GPU (permanent)        [Stream 0]
      → Produces: top_experts[8] = {expert_id, gate_weight}
   c. SHARED_EXPERT → GPU (permanent)       [Stream 0]
   d. CACHE_LOOKUP → Expert Cache           [Stream 1]
      → For each of 8 active experts:
         HIT:  compute immediately from VRAM
         MISS: issue hipMemcpyAsync from CPU RAM  [Stream 2]
   e. EXPERT_MATMUL → GPU                   [Stream 1, sync Stream 2 if miss]
      → 8 expert outputs × gate_weights
   f. COMBINE → weighted sum + shared expert [Stream 1]
   g. PREFETCH_NEXT → start loading L+1's predicted experts [Stream 2]
3. LM_HEAD → GPU (permanent)               [Stream 0]
4. SAMPLE  → CPU
```

### Key Optimizations

**1. Layer-Ahead Prefetching**
After computing layer N's routing, predict layer N+1's experts:
- Routing history (same experts tend to fire across adjacent layers)
- Static profiling (some experts are universally hot — always prefetched)
- Speculative top-16 (load 16 predicted, only need 8 — 50% slack for misprediction)

**2. Frequency-Weighted LRU Cache**
Standard LRU evicts least-recently used. We add frequency:
`score = recency_weight × time_since_last_use + (1 - recency_weight) × 1/frequency`
Hot experts that fire often never get evicted, even if a few tokens skip them.

**3. Expert Coalescing**
Experts that frequently co-fire are placed in contiguous GPU cache slots.
PCIe DMA is faster for contiguous transfers — one 18.9 MB bulk copy (8 experts)
is faster than 8 × 2.36 MB scattered copies.

**4. Per-Layer Reservation**
Reserve cache slots for each layer's "guaranteed hot" experts (top ~5 by frequency).
48 layers × 5 reserved = 240 experts = 566 MB — always warm, zero miss.

---

## Implementation Plan

### Phase 1: Expert Weight Manager (C++)
**Files**: `core/include/expertflow/`, `core/src/expertflow/`

```
expert_map.h/cpp     — Parse GGUF, build expert → offset map
expert_cache.h/cpp   — GPU LRU cache with freq-weighted eviction
expert_profiler.h/cpp — Hot expert detection via calibration
```

**expert_map**: Reads GGUF metadata, identifies which tensors are experts vs shared.
For Qwen3.5-35B-A3B:
- Shared: `model.layers.N.self_attn.*`, `model.layers.N.mlp.gate.*`, norms
- Experts: `model.layers.N.mlp.experts.E.*` (gate_proj, up_proj, down_proj)

**expert_cache**: Fixed-size GPU buffer pool.
- `allocate(n_experts, expert_size)` → pre-allocates slots
- `get(layer_id, expert_id)` → returns GPU pointer or nullptr (miss)
- `insert(layer_id, expert_id, data)` → copies to GPU, evicts if full
- Eviction: LRU with frequency boost (recently AND frequently used survive)

### Phase 2: Async Prefetcher (HIP/CUDA)
**Files**: `core/src/expertflow/`

```
expert_prefetcher.cpp   — Async H2D with stream management
pipeline_controller.cpp — Triple-buffered compute/transfer overlap
```

**prefetcher**: Manages HIP streams for overlapping transfer with compute.
- `prefetch(layer_id, expert_ids[])` → non-blocking H2D copy
- `await(layer_id, expert_ids[])` → blocks until transfer complete
- Double-buffered: while computing layer N, prefetch layer N+1

**pipeline_controller**: Orchestrates the full inference loop.
- Creates 3 HIP streams
- Manages event synchronization
- Tracks inflight transfers

### Phase 3: Fused MoE Kernel (HIP)
**Files**: `core/src/expertflow/`

```
moe_dispatch.cu — Fused routing × expert matmul × combine
```

Standard MoE: N separate matmuls + combine
Fused: single kernel launch, routing weights in registers, expert pointers as arguments

### Phase 4: ggml Integration
**Files**: Patches to `engine/llama.cpp/ggml/`

Hook into `ggml_backend_hip_compute` for MoE FFN layers:
- Intercept `GGML_OP_MUL_MAT` on expert tensors
- Redirect to ExpertFlow dispatch
- Fall back to standard path for non-MoE models

---

## File Structure

```
core/
├── include/expertflow/
│   ├── expert_map.h          — GGUF expert tensor decomposition        ✅
│   ├── expert_cache.h        — GPU-side LRU expert cache               ✅
│   ├── expert_prefetcher.h   — Async H2D pipeline                      ✅
│   ├── pipeline_controller.h — Stream orchestration                    ✅
│   └── moe_dispatch.h        — Fused MoE kernel interface              ✅
├── src/expertflow/
│   ├── expert_map.cpp         — GGUF parser (verified vs real 122B)    ✅
│   ├── expert_cache.cpp       — LRU + freq-weighted eviction           ✅
│   ├── expert_prefetcher.cpp  — Coalesced staging + layer-ahead pred   ✅
│   ├── pipeline_controller.cpp — 3-stream mmap orchestrator            ✅
│   ├── moe_dispatch.cpp       — CPU ref: dequant + SiLU + matmul      ✅
│   └── expertflow_backend.cpp — ggml integration bridge               ✅
├── tests/
│   └── test_expertflow.cpp    — 26 tests (all pass)                    ✅
├── EXPERTFLOW.md              — Research + architecture document       ✅
└── INTEGRATION.md             — llama.cpp patch guide (3 files, ~50 lines) ✅
```

### Test Inventory (35/35 pass)

| # | Test | Component |
|---|------|----------|
| 1-8 | expert_map_* | GGUF parsing, architecture, tensor counts, expert lookup, sizes, speed estimates |
| 9-14 | cache_* | Init, hit/miss, LRU eviction, frequency weighting, batch lookup, hit rate report |
| 15-17 | prefetcher_* | Init/release, submit/await, data integrity (memcmp vs mmap) |
| 18-19 | pipeline_* | Full init+process (5 tokens, 86.7% hit rate), warmup simulation (200 tokens, 60.6%) |
| 20-23 | dispatch_* | FP32 identity, multi-expert accumulation, FLOPs calculation, SiLU correctness |
| 24-26 | backend_* | Tensor classification, create+prepare (3 tokens), global instance management |
| 27-28 | **Phase A** | TQ3 KV compression ratio integration, auto_config VRAM budget |
| 29-31 | **Phase C** | Expert compression round-trip, real IQ2_XXS data (89.7% savings), pinned pool |
| 32-34 | **Phase D** | Routing predictor init, pattern learning (100% accuracy), accuracy improvement (74%→92%) |
| 35 | **Phase B** | GPU dispatch fallback (CPU when no CUDA/HIP backend) |

---

## Research References

- **MoE Offloading**: Eliseev & Mazur, "Fast Inference of Mixture-of-Experts Language Models with Offloading" (arXiv:2312.17238)
  - Expert LRU cache + speculative loading
  - Achieved 2-3x speedup on Mixtral-8x7B with consumer GPU
  - Our approach extends this with frequency-weighted eviction + layer-ahead prefetching

- **Pre-gated MoE**: Hwang et al., "Pre-gated MoE" (arXiv:2305.10601)
  - Predict expert routing 1 layer ahead
  - Accuracy within 90% of actual routing
  - Enables prefetching before routing decision is final

- **Expert Choice Routing**: Zhou et al., "Mixture-of-Experts with Expert Choice Routing" (arXiv:2202.09368)
  - Load-balanced routing improves cache locality
  - Fewer unique experts activated per batch

- **DeepSpeed-MoE**: Rajbhandari et al., "DeepSpeed-MoE" (arXiv:2201.05596)
  - Expert parallelism across devices
  - Concepts applicable to CPU-GPU split

---

## Milestones

| Phase | Deliverable | Status |
|-------|-------------|--------|
| 1 | Expert map + GPU cache (14 tests) | ✅ Complete |
| 2 | Async prefetcher + pipeline controller (20 tests) | ✅ Complete |
| 3 | Fused MoE dispatch CPU reference (23 tests) | ✅ Complete |
| 4 | ggml backend bridge + integration guide (26 tests) | ✅ Complete |
| 5 | **Turbo optimizations A-D** (35 tests) | ✅ Complete |
| 6 | llama.cpp integration + end-to-end inference | 🔄 In Progress |

## Success Criteria

- [x] GGUF parser verified against real Qwen3.5-122B-A10B model
- [x] Expert cache with LRU + frequency-weighted eviction
- [x] Async prefetcher with data integrity verification (memcmp)
- [x] Pipeline controller: mmap + 3-stream orchestration + profiling
- [x] MoE dispatch: CPU reference with correct SiLU + multi-expert accumulation
- [x] 35/35 unit tests passing
- [x] Expert cache hit rate >80% with repeated routing (86.7% measured)
- [x] Realistic warmup simulation: 60.6% hit rate with 1GB cache, 200 tokens
- [x] ggml backend bridge: tensor classification + ExpertFlowBackend + global instance
- [x] Integration guide: 3-file patch for llama.cpp (~50 lines)
- [x] Phase A: TurboQuant KV compression → +156 MB VRAM, +62 expert slots
- [x] Phase B: Fused CUDA MoE dispatch kernel (IQ2_XXS + F16 dequant GEMV)
- [x] Phase C: Expert transfer compression (89.7% savings) + pinned memory pool
- [x] Phase D: Adaptive routing predictor (Markov chain, 74%→92% accuracy)
- [ ] Apply patches to local llama.cpp and run end-to-end inference
- [ ] PCIe transfer fully overlapped with compute (GPU streams)
- [ ] Target: 57-68 tok/s generation (11× over current 5.1 tok/s)
