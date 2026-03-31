// ExpertFlow — ggml Backend Integration Bridge
// expertflow_backend.h: Hooks for integrating ExpertFlow with llama.cpp
//
// This module provides the bridge between ExpertFlow's expert cache/prefetch
// system and llama.cpp's ggml-based inference engine.
//
// Integration strategy:
//   1. Tensor placement: expert tensors stay on CPU (mmap), shared on GPU
//   2. Runtime hook: after router selects top-K experts, ExpertFlow
//      looks them up in cache and prefetches missing ones
//   3. The ggml_mul_mat_id operation uses cached GPU pointers
//
// Minimal patches to llama.cpp:
//   - llama-model.cpp: override expert tensor placement (CPU always)
//   - llama-graph.cpp: insert ExpertFlow hook in build_moe_ffn()
//   - llama.cpp: initialize ExpertFlow during model load

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "expertflow/expert_cache.h"
#include "expertflow/expert_map.h"
#include "expertflow/expert_prefetcher.h"
#include "expertflow/pipeline_controller.h"

namespace expertflow {

// ============================================================
// Tensor Placement Policy
// ============================================================

// Determines where a tensor should be placed (GPU or CPU).
// Used to override llama.cpp's default ngl-based placement.
enum class TensorPlacement : uint8_t {
    kGPU     = 0,  // Tensor stays on GPU (shared weights)
    kCPU     = 1,  // Tensor stays on CPU (expert weights, will be cached)
    kDefault = 2,  // Use llama.cpp's default placement
};

// Given a tensor name, decide where it should live.
// This is the core of ExpertFlow's placement strategy:
//   - Expert projections (ffn_gate_exps, ffn_up_exps, ffn_down_exps) → CPU
//   - Everything else (attention, norms, router, embeddings) → GPU
//
// Call this from llama-model.cpp's create_tensor to override the buft_list.
TensorPlacement classify_tensor(const std::string& tensor_name);

// Check if a tensor name is an expert weight (should be on CPU for ExpertFlow)
bool is_expert_tensor(const std::string& tensor_name);

// Check if a tensor name is a shared weight (should be on GPU permanently)
bool is_shared_tensor(const std::string& tensor_name);

// ============================================================
// ExpertFlow Session — per-model instance
// ============================================================

// Configuration for ExpertFlow integration
struct BackendConfig {
    bool     enabled;                // Master enable switch
    size_t   expert_cache_bytes;     // VRAM budget for expert cache
    size_t   staging_buffer_bytes;   // Pinned host staging buffer
    uint32_t speculative_top_k;     // Speculative prefetch count
    uint32_t reserved_hot_per_layer; // Hot experts pinned per layer
    float    recency_weight;         // LRU vs frequency balance
    bool     enable_prefetch;        // Layer-ahead prefetching
    bool     enable_coalescing;      // DMA coalescing
    bool     enable_profiling;       // Per-token profiling

    // Generic defaults (model-agnostic, conservative cache size)
    static BackendConfig defaults() {
        return BackendConfig{
            .enabled                = true,
            .expert_cache_bytes     = 1ULL * 1024 * 1024 * 1024,  // 1 GB
            .staging_buffer_bytes   = 64ULL * 1024 * 1024,         // 64 MB
            .speculative_top_k      = 12,
            .reserved_hot_per_layer = 5,
            .recency_weight         = 0.6f,
            .enable_prefetch        = true,
            .enable_coalescing      = true,
            .enable_profiling       = false,
        };
    }

    // Auto-configure from available VRAM and model architecture.
    // Computes expert cache size by subtracting shared weight + KV overhead.
    //   vram_bytes: total GPU VRAM (0 = use defaults)
    //   arch: model architecture (from ExpertMap)
    //   kv_compression_ratio: KV cache bytes after compression / bytes before
    //     1.0 = no compression (FP16 KV), ~0.22 = TQ3 (3.5/16), ~0.16 = TQ2
    static BackendConfig auto_config(size_t vram_bytes,
                                     const MoeArchitecture& arch,
                                     float kv_compression_ratio = 1.0f) {
        auto cfg = defaults();
        if (vram_bytes == 0) return cfg;

        // KV overhead: 200 MB baseline scaled by compression ratio
        size_t kv_overhead = static_cast<size_t>(
            200ULL * 1024 * 1024 * std::max(0.1f, kv_compression_ratio));

        // Reserve space for shared weights + compressed KV cache
        size_t reserved = arch.shared_weight_bytes + kv_overhead;
        if (vram_bytes > reserved) {
            cfg.expert_cache_bytes = vram_bytes - reserved;
        } else {
            // Minimal: 2 × top-K experts (enough for one layer pair)
            cfg.expert_cache_bytes = 2 * arch.n_experts_used *
                                     arch.expert_weight_bytes;
        }

        // Staging: 2 × top-K experts for double buffering
        if (arch.expert_weight_bytes > 0 && arch.n_experts_used > 0) {
            size_t min_staging = 2 * arch.n_experts_used *
                                 arch.expert_weight_bytes;
            cfg.staging_buffer_bytes = std::max(cfg.staging_buffer_bytes,
                                                min_staging);
        }

        // Speculative: 50% extra over top-K
        if (arch.n_experts_used > 0) {
            cfg.speculative_top_k = arch.n_experts_used +
                                    arch.n_experts_used / 2;
        }

        return cfg;
    }
};

// ExpertFlow backend session — manages the lifecycle of ExpertFlow
// for a single model instance.
//
// Usage from llama.cpp:
//   1. After model loading: ExpertFlowBackend::create(model_path, config)
//   2. Before inference: session->begin_token()
//   3. In build_moe_ffn, after router: session->prepare_experts(layer, expert_ids)
//   4. After token: session->end_token()
//
class ExpertFlowBackend {
public:
    ~ExpertFlowBackend();

    // Create an ExpertFlow session for a model.
    // Returns nullptr if the model is not MoE or initialization fails.
    static std::unique_ptr<ExpertFlowBackend> create(
        const std::string& gguf_path,
        const BackendConfig& config = BackendConfig::defaults());

    // Is ExpertFlow active for this model?
    bool is_active() const { return pipeline_.is_ready(); }

    // Get the model architecture info
    const MoeArchitecture& architecture() const {
        return pipeline_.expert_map().architecture();
    }

    // --- Per-token lifecycle ---

    // Call at the start of each token generation
    void begin_token();

    // Prepare experts for a layer's MoE block.
    // Called after the router has produced selected_experts.
    //
    // Parameters:
    //   layer_id: transformer layer index (0..n_layers-1)
    //   expert_ids: top-K expert indices from the router
    //   gate_weights: corresponding gating weights
    //
    // Returns: GPU pointers for each expert's 3 projections,
    //          ready for use in ggml_mul_mat_id.
    //
    // This function:
    //   1. Looks up experts in the GPU cache
    //   2. Issues async H2D for cache misses
    //   3. Waits for transfers to complete
    //   4. Returns valid GPU pointers
    //   5. Submits prefetch for the next layer
    PipelineController::ExpertPointers prepare_experts(
        uint32_t layer_id,
        const std::vector<uint32_t>& expert_ids,
        const std::vector<float>& gate_weights);

    // Call at the end of each token generation
    void end_token();

    // --- Profiling ---

    // Get pipeline statistics
    PipelineStats stats() const { return pipeline_.compute_stats(); }

    // Print full performance report
    void print_report() const { pipeline_.print_report(); }

    // Get the underlying pipeline (for advanced use)
    const PipelineController& pipeline() const { return pipeline_; }

private:
    ExpertFlowBackend() = default;

    PipelineController pipeline_;
    BackendConfig config_{};
};

// ============================================================
// Global ExpertFlow instance management
// ============================================================

// Set the global ExpertFlow backend for the current model.
// Called during model loading.
void set_global_backend(std::unique_ptr<ExpertFlowBackend> backend);

// Get the global ExpertFlow backend (may be nullptr if not active).
ExpertFlowBackend* get_global_backend();

// Check if ExpertFlow is active for the current model.
bool is_expertflow_active();

}  // namespace expertflow
