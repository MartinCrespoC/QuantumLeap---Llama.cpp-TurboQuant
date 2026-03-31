# ExpertFlow — llama.cpp Integration Guide

## Overview

ExpertFlow integrates with llama.cpp by overriding **tensor placement** and **expert dispatch**.
Instead of offloading entire layers (all 256 experts) via `ngl`, ExpertFlow keeps expert weights
on CPU and streams only the 8 active experts per layer to GPU on demand.

**Minimal changes**: 3 files modified in `engine/llama.cpp/src/`, ~50 lines total.

---

## Patch Points

### 1. Model Loading: `llama-model.cpp`

**Goal**: Force expert tensors to CPU, shared tensors to GPU.

In `load_tensors()`, after `get_layer_buft_list` is defined (~line 2644), wrap the
`create_tensor` lambda to check ExpertFlow placement:

```cpp
// === ExpertFlow: override expert tensor placement ===
#include "expertflow/expertflow_backend.h"

auto create_tensor_ef = [&](const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags) -> ggml_tensor * {
    // If ExpertFlow is active, force expert tensors to CPU
    if (expertflow::is_expertflow_active()) {
        auto placement = expertflow::classify_tensor(tn.str());
        if (placement == expertflow::TensorPlacement::kCPU) {
            // Use CPU buffer list regardless of ngl setting
            return ml.create_tensor(
                hparams, &pimpl->cpu_buft_list, pimpl->dev_input.buft_list,
                pimpl->dev_output.buft_list, &pimpl->cpu_buft_list,
                tn, ne, flags);
        }
        if (placement == expertflow::TensorPlacement::kGPU && !devices.empty()) {
            // Force shared tensors to GPU even if ngl=0
            auto * dev = devices.at(0);
            return ml.create_tensor(
                hparams, &pimpl->cpu_buft_list, pimpl->dev_input.buft_list,
                pimpl->dev_output.buft_list, &pimpl->gpu_buft_list.at(dev),
                tn, ne, flags);
        }
    }
    // Default: use original create_tensor
    return ml.create_tensor(
        hparams, &pimpl->cpu_buft_list, pimpl->dev_input.buft_list,
        pimpl->dev_output.buft_list, tn.bid == -1 ? nullptr : pimpl->dev_layer.at(tn.bid).buft_list,
        tn, ne, flags);
};
```

Then replace `create_tensor` calls with `create_tensor_ef` in the model-specific switch cases.

### 2. Inference Hook: `llama-graph.cpp`

**Goal**: After router produces `selected_experts`, intercept and prepare GPU pointers.

In `build_moe_ffn()`, after `selected_experts` is computed (~line 1307), add:

```cpp
// === ExpertFlow: prefetch active experts ===
if (expertflow::is_expertflow_active()) {
    auto* ef = expertflow::get_global_backend();
    // Extract selected expert IDs (will be resolved at graph eval time)
    // For now, submit prefetch hint based on layer index
    // The actual expert resolution happens in the ggml backend
    // via a custom op or callback
}
```

**Note**: Full integration requires either:
- A custom ggml op that calls ExpertFlow at eval time (preferred)
- A graph execution callback that intercepts `ggml_mul_mat_id` on expert tensors

### 3. Session Lifecycle: `llama.cpp`

**Goal**: Initialize ExpertFlow during model load, hook token lifecycle.

In `llama_model_load()`:
```cpp
// After model tensors are loaded:
if (hparams.n_expert > 0) {
    auto ef_config = expertflow::BackendConfig::default_config();
    auto ef = expertflow::ExpertFlowBackend::create(path_model, ef_config);
    if (ef) {
        expertflow::set_global_backend(std::move(ef));
    }
}
```

In `llama_decode_impl()`:
```cpp
if (expertflow::is_expertflow_active()) {
    expertflow::get_global_backend()->begin_token();
}
// ... existing decode logic ...
if (expertflow::is_expertflow_active()) {
    expertflow::get_global_backend()->end_token();
}
```

---

## Build Integration

Add to `engine/llama.cpp/CMakeLists.txt`:

```cmake
# ExpertFlow integration
option(LLAMA_EXPERTFLOW "Enable ExpertFlow MoE optimization" OFF)
if (LLAMA_EXPERTFLOW)
    add_subdirectory(../../core core_build)
    target_link_libraries(llama PRIVATE expertflow)
    target_compile_definitions(llama PRIVATE LLAMA_EXPERTFLOW=1)
endif()
```

---

## Architecture: How it Works

```
┌─ llama.cpp inference loop ──────────────────────────────┐
│                                                          │
│  1. Token embed → GPU                                    │
│  2. For each layer:                                      │
│     a. Attention (shared weights, always on GPU)         │
│     b. Router: gate_inp × cur → selected_experts         │
│     c. ── ExpertFlow Hook ──                             │
│        │  prepare_experts(layer, selected_experts)        │
│        │  → cache lookup → async H2D for misses          │
│        │  → prefetch next layer's predicted experts      │
│        │  → return GPU pointers                          │
│        └─────────────────────────────────────────────    │
│     d. Expert matmul using cached GPU pointers           │
│     e. Weighted sum + residual                           │
│  3. LM head → sample                                    │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## API Reference

### Tensor Classification
```cpp
expertflow::classify_tensor(name)    → kGPU / kCPU / kDefault
expertflow::is_expert_tensor(name)   → bool (ffn_*_exps)
expertflow::is_shared_tensor(name)   → bool (everything else)
```

### Backend Lifecycle
```cpp
auto backend = ExpertFlowBackend::create(path, config);
set_global_backend(std::move(backend));

// Per-token:
get_global_backend()->begin_token();
auto ptrs = get_global_backend()->prepare_experts(layer, ids, weights);
// use ptrs.gate_ptrs, ptrs.up_ptrs, ptrs.down_ptrs for matmul
get_global_backend()->end_token();
```

### Configuration
```cpp
BackendConfig cfg = BackendConfig::default_config();
cfg.expert_cache_bytes     = 1900 * 1024 * 1024;  // 1.9 GB VRAM
cfg.staging_buffer_bytes   = 64 * 1024 * 1024;     // 64 MB pinned
cfg.speculative_top_k      = 12;                    // prefetch 12, need 8
cfg.reserved_hot_per_layer = 5;                     // pin top-5 per layer
cfg.recency_weight         = 0.6f;                  // 60% LRU, 40% freq
```
