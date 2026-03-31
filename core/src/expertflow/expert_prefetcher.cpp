// ExpertFlow — expert_prefetcher.cpp
// Async GPU expert prefetching with HIP/CUDA/CPU-fallback streams
//
// Overlaps H2D expert weight transfer with GPU computation using
// a dedicated transfer stream. Supports layer-ahead prediction
// and coalesced DMA for contiguous expert blocks.
//
// PCIe budget: 25 GB/s real → 2.51 MB per expert → 0.100 ms per expert
// Target: hide all PCIe latency behind 0.247 ms attention compute per layer

#include "expertflow/expert_prefetcher.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <numeric>

// GPU API abstraction
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#define GPU_STREAM_CREATE(s)         hipStreamCreateWithFlags((s), hipStreamNonBlocking)
#define GPU_STREAM_DESTROY(s)        hipStreamDestroy((s))
#define GPU_STREAM_SYNC(s)           hipStreamSynchronize((s))
#define GPU_EVENT_CREATE(e)          hipEventCreateWithFlags((e), hipEventDisableTiming)
#define GPU_EVENT_DESTROY(e)         hipEventDestroy((e))
#define GPU_EVENT_RECORD(e, s)       hipEventRecord((e), (s))
#define GPU_EVENT_QUERY(e)           hipEventQuery((e))
#define GPU_EVENT_SYNC(e)            hipEventSynchronize((e))
#define GPU_MEMCPY_ASYNC(dst, src, n, s) hipMemcpyAsync((dst), (src), (n), hipMemcpyHostToDevice, (s))
#define GPU_MALLOC_HOST(ptr, sz)     hipHostMalloc((ptr), (sz), hipHostMallocDefault)
#define GPU_FREE_HOST(ptr)           hipHostFree((ptr))
#define GPU_SUCCESS                  hipSuccess
#elif defined(__CUDACC__) || defined(EXPERTFLOW_CUDA)
#include <cuda_runtime.h>
#define GPU_STREAM_CREATE(s)         cudaStreamCreateWithFlags((s), cudaStreamNonBlocking)
#define GPU_STREAM_DESTROY(s)        cudaStreamDestroy((s))
#define GPU_STREAM_SYNC(s)           cudaStreamSynchronize((s))
#define GPU_EVENT_CREATE(e)          cudaEventCreateWithFlags((e), cudaEventDisableTiming)
#define GPU_EVENT_DESTROY(e)         cudaEventDestroy((e))
#define GPU_EVENT_RECORD(e, s)       cudaEventRecord((e), (s))
#define GPU_EVENT_QUERY(e)           cudaEventQuery((e))
#define GPU_EVENT_SYNC(e)            cudaEventSynchronize((e))
#define GPU_MEMCPY_ASYNC(dst, src, n, s) cudaMemcpyAsync((dst), (src), (n), cudaMemcpyHostToDevice, (s))
#define GPU_MALLOC_HOST(ptr, sz)     cudaMallocHost((ptr), (sz))
#define GPU_FREE_HOST(ptr)           cudaFreeHost((ptr))
#define GPU_SUCCESS                  cudaSuccess
#else
// CPU fallback: synchronous memcpy, no streams
#include <cstdlib>
#define GPU_STREAM_CREATE(s)         (*(s) = nullptr, 0)
#define GPU_STREAM_DESTROY(s)        ((void)0)
#define GPU_STREAM_SYNC(s)           ((void)0)
#define GPU_EVENT_CREATE(e)          (*(e) = nullptr, 0)
#define GPU_EVENT_DESTROY(e)         ((void)0)
#define GPU_EVENT_RECORD(e, s)       ((void)0)
#define GPU_EVENT_QUERY(e)           (0)  // always "complete"
#define GPU_EVENT_SYNC(e)            (0)  // no-op, always success
#define GPU_MEMCPY_ASYNC(dst, src, n, s) (memcpy((dst), (src), (n)), 0)
#define GPU_MALLOC_HOST(ptr, sz)     (*(ptr) = static_cast<uint8_t*>(malloc(sz)), 0)
#define GPU_FREE_HOST(ptr)           free((ptr))
#define GPU_SUCCESS                  0
#endif

namespace expertflow {

ExpertPrefetcher::~ExpertPrefetcher() {
    release();
}

bool ExpertPrefetcher::init(const PrefetchConfig& config,
                             ExpertCache* cache,
                             const ExpertMap* map,
                             const uint8_t* cpu_base) {
    release();

    config_   = config;
    cache_    = cache;
    map_      = map;
    cpu_base_ = cpu_base;
    stats_    = {};

    if (!cache_ || !map_ || !cpu_base_) {
        fprintf(stderr, "[ExpertPrefetcher] Null cache, map, or cpu_base\n");
        return false;
    }

    // Create dedicated transfer stream
    auto err = GPU_STREAM_CREATE(&transfer_stream_);
    if (err != GPU_SUCCESS) {
        fprintf(stderr, "[ExpertPrefetcher] Failed to create transfer stream\n");
        return false;
    }

    // Create completion event
    err = GPU_EVENT_CREATE(&transfer_done_);
    if (err != GPU_SUCCESS) {
        fprintf(stderr, "[ExpertPrefetcher] Failed to create transfer event\n");
        GPU_STREAM_DESTROY(transfer_stream_);
        transfer_stream_ = nullptr;
        return false;
    }

    // Allocate pinned host staging buffer for faster H2D
    if (config.staging_buffer_bytes > 0) {
        err = GPU_MALLOC_HOST(&staging_buffer_, config.staging_buffer_bytes);
        if (err != GPU_SUCCESS) {
            fprintf(stderr, "[ExpertPrefetcher] Failed to alloc staging buffer (%zu bytes)\n",
                    config.staging_buffer_bytes);
            // Non-fatal: we can still do direct mmap → GPU copies (slower)
            staging_buffer_ = nullptr;
            staging_bytes_ = 0;
        } else {
            staging_bytes_ = config.staging_buffer_bytes;
        }
    }

    // Set default routing predictor if none provided
    if (!predictor_) {
        predictor_ = default_routing_predictor;
    }

    current_status_ = TransferStatus::kIdle;

    printf("[ExpertPrefetcher] Initialized: stream=%p, staging=%zu MB, prefetch=%s\n",
           (void*)transfer_stream_,
           staging_bytes_ / (1024 * 1024),
           config.enable_prefetch ? "ON" : "OFF");

    return true;
}

void ExpertPrefetcher::release() {
    if (transfer_stream_) {
        GPU_STREAM_SYNC(transfer_stream_);
        GPU_STREAM_DESTROY(transfer_stream_);
        transfer_stream_ = nullptr;
    }
    if (transfer_done_) {
        GPU_EVENT_DESTROY(transfer_done_);
        transfer_done_ = nullptr;
    }
    if (staging_buffer_) {
        GPU_FREE_HOST(staging_buffer_);
        staging_buffer_ = nullptr;
        staging_bytes_ = 0;
    }
    current_status_ = TransferStatus::kIdle;
}

void ExpertPrefetcher::submit_prefetch(uint32_t layer_id,
                                        const std::vector<uint32_t>& expert_ids) {
    if (!config_.enable_prefetch || !cache_ || !map_) return;

    // Build transfer requests for experts not in cache
    auto requests = build_requests(layer_id, expert_ids);

    if (requests.empty()) {
        // All already cached — no transfer needed
        current_status_ = TransferStatus::kComplete;
        return;
    }

    // Issue async copies
    issue_transfers(requests);
}

void ExpertPrefetcher::submit_predicted_prefetch(
    uint32_t current_layer,
    const std::vector<uint32_t>& current_expert_ids) {

    if (!config_.enable_layer_ahead || !predictor_) return;

    uint32_t next_layer = current_layer + 1;
    if (next_layer >= map_->architecture().n_layers) return;

    // Predict next layer's experts
    auto predicted = predictor_(current_layer, current_expert_ids);

    // Optionally expand to speculative_top_k
    if (config_.speculative_top_k > 0 &&
        predicted.size() < config_.speculative_top_k) {
        // Pad with frequently-accessed experts for this layer
        // (simple heuristic: use same experts as current)
        for (uint32_t eid : current_expert_ids) {
            if (predicted.size() >= config_.speculative_top_k) break;
            if (std::find(predicted.begin(), predicted.end(), eid) == predicted.end()) {
                predicted.push_back(eid);
            }
        }
    }

    submit_prefetch(next_layer, predicted);
}

bool ExpertPrefetcher::await_prefetch() {
    if (current_status_ == TransferStatus::kIdle ||
        current_status_ == TransferStatus::kComplete) {
        return true;
    }

    // Wait for the transfer event to complete
    auto err = GPU_EVENT_SYNC(transfer_done_);
    if (err != GPU_SUCCESS) {
        fprintf(stderr, "[ExpertPrefetcher] Transfer sync failed\n");
        stats_.pipeline_stalls++;
        current_status_ = TransferStatus::kIdle;
        return false;
    }

    current_status_ = TransferStatus::kComplete;
    return true;
}

TransferStatus ExpertPrefetcher::query_status() const {
    if (current_status_ != TransferStatus::kInProgress) {
        return current_status_;
    }

    // Non-blocking query
    auto err = GPU_EVENT_QUERY(transfer_done_);
    if (err == GPU_SUCCESS) {
        return TransferStatus::kComplete;
    }
    return TransferStatus::kInProgress;
}

void ExpertPrefetcher::sync() {
    if (transfer_stream_) {
        GPU_STREAM_SYNC(transfer_stream_);
    }
    current_status_ = TransferStatus::kIdle;
}

void ExpertPrefetcher::set_predictor(RoutingPredictor predictor) {
    predictor_ = std::move(predictor);
}

std::vector<TransferRequest> ExpertPrefetcher::build_requests(
    uint32_t layer_id,
    const std::vector<uint32_t>& expert_ids) {

    std::vector<TransferRequest> requests;
    requests.reserve(expert_ids.size() * static_cast<uint32_t>(ExpertProj::kCount));

    for (uint32_t eid : expert_ids) {
        for (uint32_t p = 0; p < static_cast<uint32_t>(ExpertProj::kCount); ++p) {
            auto proj = static_cast<ExpertProj>(p);

            stats_.total_prefetches++;

            // Check cache — is this expert already on GPU?
            CacheLookup lookup = cache_->lookup(layer_id, eid, proj);

            if (lookup.result == CacheResult::kHit) {
                stats_.prefetches_hit++;
                continue;  // Already cached, no transfer needed
            }

            // Cache MISS — need to transfer from CPU
            const ExpertSlice* slice = map_->get_expert(layer_id, eid, proj);
            if (!slice) {
                fprintf(stderr, "[ExpertPrefetcher] Expert not found: layer=%u, id=%u, proj=%u\n",
                        layer_id, eid, p);
                continue;
            }

            TransferRequest req;
            req.layer_id   = layer_id;
            req.expert_id  = eid;
            req.proj       = proj;
            req.src_cpu    = cpu_base_ + slice->offset;
            req.dst_gpu    = lookup.gpu_ptr;  // Cache allocated a slot for us
            req.size_bytes = slice->size_bytes;

            requests.push_back(req);
            stats_.prefetches_issued++;
        }
    }

    return requests;
}

void ExpertPrefetcher::issue_transfers(const std::vector<TransferRequest>& requests) {
    if (requests.empty()) {
        current_status_ = TransferStatus::kComplete;
        return;
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    current_batch_bytes_ = 0;

    if (config_.enable_coalescing && staging_buffer_ && staging_bytes_ > 0) {
        // Coalesced transfer: copy all experts to pinned staging buffer first,
        // then do a single large DMA transfer to GPU.
        // This is faster than many small scattered copies because:
        // 1. Pinned memory enables DMA (bypass CPU cache)
        // 2. Single large transfer has less PCIe overhead than many small ones

        size_t total_bytes = 0;
        for (const auto& req : requests) total_bytes += req.size_bytes;

        if (total_bytes <= staging_bytes_) {
            // Stage all experts into contiguous pinned buffer
            size_t offset = 0;
            for (const auto& req : requests) {
                memcpy(staging_buffer_ + offset, req.src_cpu, req.size_bytes);
                offset += req.size_bytes;
            }

            // Now copy from staging to each GPU slot
            // (We still need per-slot copies since GPU slots aren't contiguous)
            offset = 0;
            for (const auto& req : requests) {
                GPU_MEMCPY_ASYNC(req.dst_gpu, staging_buffer_ + offset,
                                 req.size_bytes, transfer_stream_);
                offset += req.size_bytes;
                current_batch_bytes_ += req.size_bytes;
            }
        } else {
            // Staging buffer too small — fall back to direct copies
            for (const auto& req : requests) {
                GPU_MEMCPY_ASYNC(req.dst_gpu, req.src_cpu,
                                 req.size_bytes, transfer_stream_);
                current_batch_bytes_ += req.size_bytes;
            }
        }
    } else {
        // Direct mmap → GPU async copy (no staging)
        // This still works but may be slower if source isn't pinned
        for (const auto& req : requests) {
            GPU_MEMCPY_ASYNC(req.dst_gpu, req.src_cpu,
                             req.size_bytes, transfer_stream_);
            current_batch_bytes_ += req.size_bytes;
        }
    }

    // Record event to signal completion
    GPU_EVENT_RECORD(transfer_done_, transfer_stream_);
    current_status_ = TransferStatus::kInProgress;

    stats_.bytes_transferred += current_batch_bytes_;

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    // Update rolling average
    double total_time = stats_.avg_transfer_ms * (stats_.prefetches_issued - requests.size())
                      + elapsed_ms;
    if (stats_.prefetches_issued > 0) {
        stats_.avg_transfer_ms = total_time / stats_.prefetches_issued;
    }
}

void ExpertPrefetcher::reset_stats() {
    stats_ = {};
}

void ExpertPrefetcher::print_status() const {
    printf("=== ExpertPrefetcher Status ===\n");
    printf("Prefetch: %s, Layer-ahead: %s, Coalescing: %s\n",
           config_.enable_prefetch ? "ON" : "OFF",
           config_.enable_layer_ahead ? "ON" : "OFF",
           config_.enable_coalescing ? "ON" : "OFF");
    printf("Total prefetches: %lu (hit: %lu, issued: %lu)\n",
           stats_.total_prefetches, stats_.prefetches_hit, stats_.prefetches_issued);
    printf("Prefetch efficiency: %.1f%% (already cached)\n",
           stats_.prefetch_efficiency() * 100.0);
    printf("Bytes transferred: %.1f MB\n",
           stats_.bytes_transferred / (1024.0 * 1024.0));
    printf("Pipeline stalls: %lu\n", stats_.pipeline_stalls);
    printf("Avg transfer time: %.3f ms\n", stats_.avg_transfer_ms);
    printf("Staging buffer: %zu MB\n", staging_bytes_ / (1024 * 1024));
    printf("=== End ExpertPrefetcher ===\n");
}

// Default routing predictor: assume next layer uses the same experts
std::vector<uint32_t> default_routing_predictor(
    uint32_t /*current_layer*/,
    const std::vector<uint32_t>& current_expert_ids) {
    return current_expert_ids;
}

}  // namespace expertflow
