// ExpertFlow — MoE-Aware Inference Engine
// expert_prefetcher.h: Async GPU expert prefetching with HIP/CUDA streams
//
// Manages asynchronous H2D transfers to overlap expert weight loading
// with GPU computation. Uses double-buffered staging to hide PCIe latency.
//
// Pipeline:
//   Layer N compute (Stream 1) || Layer N+1 expert prefetch (Stream 2)
//
// Target: PCIe 4.0 x16 = 25 GB/s real-world
// Per expert: 2.36 MB → 0.094 ms per expert transfer
// 8 experts per layer: 0.755 ms if sequential, overlapped with compute

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "expertflow/expert_cache.h"
#include "expertflow/expert_map.h"

// Forward-declare GPU stream handle (HIP or CUDA)
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
using GpuStream_t = hipStream_t;
using GpuEvent_t  = hipEvent_t;
#elif defined(__CUDACC__) || defined(EXPERTFLOW_CUDA)
#include <cuda_runtime.h>
using GpuStream_t = cudaStream_t;
using GpuEvent_t  = cudaEvent_t;
#else
// CPU-only fallback: streams are no-ops
using GpuStream_t = void*;
using GpuEvent_t  = void*;
#endif

namespace expertflow {

// Transfer request for a single expert
struct TransferRequest {
    uint32_t    layer_id;
    uint32_t    expert_id;
    ExpertProj  proj;
    const uint8_t* src_cpu;    // Source: CPU RAM (mmap'd GGUF)
    uint8_t*       dst_gpu;    // Destination: GPU cache slot
    size_t         size_bytes; // Transfer size
};

// Status of an inflight transfer batch
enum class TransferStatus : uint8_t {
    kIdle       = 0,  // No transfer in progress
    kInProgress = 1,  // Async copy issued, not yet complete
    kComplete   = 2,  // Transfer finished, experts ready for compute
};

// Prefetch statistics
struct PrefetchStats {
    uint64_t total_prefetches;      // Total prefetch requests
    uint64_t prefetches_hit;        // Already in cache (no transfer needed)
    uint64_t prefetches_issued;     // Actual H2D transfers issued
    uint64_t bytes_transferred;     // Total bytes moved CPU→GPU
    uint64_t pipeline_stalls;       // Times compute had to wait for transfer
    double   avg_transfer_ms;       // Average transfer time per batch

    double prefetch_efficiency() const {
        return total_prefetches > 0
            ? static_cast<double>(prefetches_hit) /
              static_cast<double>(total_prefetches)
            : 0.0;
    }
};

// Configuration for the prefetcher
struct PrefetchConfig {
    bool     enable_prefetch;        // Master switch
    bool     enable_layer_ahead;     // Predict next layer's experts
    uint32_t speculative_top_k;      // Load top-K predicted (>= n_experts_used)
    bool     enable_coalescing;      // Coalesce contiguous expert transfers
    size_t   staging_buffer_bytes;   // Double-buffer staging area on GPU
};

// Expert routing prediction callback.
// Given the current layer's routing result, predict next layer's top-K experts.
// Default: repeat the same expert IDs (works surprisingly well for MoE).
using RoutingPredictor = std::function<std::vector<uint32_t>(
    uint32_t current_layer,
    const std::vector<uint32_t>& current_expert_ids)>;

// Async Expert Prefetcher — overlaps H2D transfer with GPU compute
//
// Usage:
//   1. Create with config + cache + map references
//   2. Call init() to create GPU streams and staging buffers
//   3. Per token, per layer:
//      a. submit_prefetch(layer+1, predicted_experts)  // start async load
//      b. ... compute layer N on Stream 1 ...
//      c. await_prefetch()                             // ensure layer+1 ready
//      d. submit_prefetch(layer+2, predicted_experts)  // pipeline continues
//   4. Call sync() at end of token to flush all pending transfers
//
class ExpertPrefetcher {
public:
    ExpertPrefetcher() = default;
    ~ExpertPrefetcher();

    // Non-copyable
    ExpertPrefetcher(const ExpertPrefetcher&) = delete;
    ExpertPrefetcher& operator=(const ExpertPrefetcher&) = delete;

    // Initialize prefetcher with GPU streams.
    // cache: the ExpertCache to load experts into
    // map: the ExpertMap for CPU-side weight locations
    // cpu_base: base pointer of mmap'd GGUF data section
    bool init(const PrefetchConfig& config,
              ExpertCache* cache,
              const ExpertMap* map,
              const uint8_t* cpu_base);

    // Release GPU streams and staging buffers
    void release();

    // Submit async prefetch for a set of experts at a given layer.
    // Checks cache first — only transfers missing experts.
    // Non-blocking: returns immediately after issuing hipMemcpyAsync.
    void submit_prefetch(uint32_t layer_id,
                         const std::vector<uint32_t>& expert_ids);

    // Submit prefetch using the routing predictor to guess next layer's experts.
    void submit_predicted_prefetch(
        uint32_t current_layer,
        const std::vector<uint32_t>& current_expert_ids);

    // Wait for the most recent prefetch batch to complete.
    // Call before computing with the prefetched experts.
    // Returns true if all experts are ready, false on error.
    bool await_prefetch();

    // Check if the current prefetch batch is complete (non-blocking).
    TransferStatus query_status() const;

    // Synchronize all pending transfers (call at end of token).
    void sync();

    // Set the routing predictor function
    void set_predictor(RoutingPredictor predictor);

    // Get the transfer stream (for synchronizing with compute streams)
    GpuStream_t transfer_stream() const { return transfer_stream_; }

    // Statistics
    const PrefetchStats& stats() const { return stats_; }
    void reset_stats();

    // Print prefetch performance summary
    void print_status() const;

private:
    // Build transfer requests for missing experts
    std::vector<TransferRequest> build_requests(
        uint32_t layer_id,
        const std::vector<uint32_t>& expert_ids);

    // Issue async copies for a batch of requests
    void issue_transfers(const std::vector<TransferRequest>& requests);

    PrefetchConfig  config_{};
    PrefetchStats   stats_{};
    ExpertCache*    cache_    = nullptr;
    const ExpertMap* map_     = nullptr;
    const uint8_t*  cpu_base_ = nullptr;

    // GPU resources
    GpuStream_t transfer_stream_ = nullptr;
    GpuEvent_t  transfer_done_   = nullptr;

    // Staging buffer (pinned host memory for faster H2D)
    uint8_t* staging_buffer_ = nullptr;
    size_t   staging_bytes_  = 0;

    // Routing predictor
    RoutingPredictor predictor_;

    // Current transfer batch state
    TransferStatus current_status_ = TransferStatus::kIdle;
    size_t         current_batch_bytes_ = 0;
};

// Default routing predictor: repeat current layer's experts for next layer.
// Empirically achieves ~60-70% hit rate on Qwen MoE models.
std::vector<uint32_t> default_routing_predictor(
    uint32_t current_layer,
    const std::vector<uint32_t>& current_expert_ids);

}  // namespace expertflow
