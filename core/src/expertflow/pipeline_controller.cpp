// ExpertFlow — pipeline_controller.cpp
// Orchestrates the full MoE inference pipeline with 3-stream overlap:
//   Stream 0: Attention + Router + Shared Expert (permanent GPU tensors)
//   Stream 1: Expert matmul from cache (active experts)
//   Stream 2: Expert H2D prefetch (next layer's experts)
//
// This is the top-level controller that ties together ExpertMap,
// ExpertCache, and ExpertPrefetcher into a working inference pipeline.

#include "expertflow/pipeline_controller.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>

// mmap for GGUF file access
#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#else
#include <io.h>
#include <windows.h>
#endif

// GPU API abstraction (same as prefetcher)
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#define GPU_STREAM_CREATE(s)      hipStreamCreateWithFlags((s), hipStreamNonBlocking)
#define GPU_STREAM_DESTROY(s)     hipStreamDestroy((s))
#define GPU_STREAM_SYNC(s)        hipStreamSynchronize((s))
#define GPU_EVENT_CREATE(e)       hipEventCreateWithFlags((e), hipEventDisableTiming)
#define GPU_EVENT_DESTROY(e)      hipEventDestroy((e))
#define GPU_EVENT_RECORD(e, s)    hipEventRecord((e), (s))
#define GPU_EVENT_SYNC(e)         hipEventSynchronize((e))
#define GPU_SUCCESS               hipSuccess
#elif defined(__CUDACC__) || defined(EXPERTFLOW_CUDA)
#include <cuda_runtime.h>
#define GPU_STREAM_CREATE(s)      cudaStreamCreateWithFlags((s), cudaStreamNonBlocking)
#define GPU_STREAM_DESTROY(s)     cudaStreamDestroy((s))
#define GPU_STREAM_SYNC(s)        cudaStreamSynchronize((s))
#define GPU_EVENT_CREATE(e)       cudaEventCreateWithFlags((e), cudaEventDisableTiming)
#define GPU_EVENT_DESTROY(e)      cudaEventDestroy((e))
#define GPU_EVENT_RECORD(e, s)    cudaEventRecord((e), (s))
#define GPU_EVENT_SYNC(e)         cudaEventSynchronize((e))
#define GPU_SUCCESS               cudaSuccess
#else
// CPU fallback
#define GPU_STREAM_CREATE(s)      (*(s) = nullptr, 0)
#define GPU_STREAM_DESTROY(s)     ((void)0)
#define GPU_STREAM_SYNC(s)        ((void)0)
#define GPU_EVENT_CREATE(e)       (*(e) = nullptr, 0)
#define GPU_EVENT_DESTROY(e)      ((void)0)
#define GPU_EVENT_RECORD(e, s)    ((void)0)
#define GPU_EVENT_SYNC(e)         ((void)0)
#define GPU_SUCCESS               0
#endif

namespace expertflow {

PipelineController::~PipelineController() {
    release();
}

bool PipelineController::init(const std::string& gguf_path,
                               const PipelineConfig& config) {
    release();
    config_ = config;

    // Step 1: Parse GGUF model
    printf("[Pipeline] Step 1/5: Parsing GGUF model...\n");
    if (!map_.load(gguf_path)) {
        fprintf(stderr, "[Pipeline] Failed to parse GGUF: %s\n", gguf_path.c_str());
        return false;
    }

    if (!map_.is_moe()) {
        fprintf(stderr, "[Pipeline] Not a MoE model — ExpertFlow not needed\n");
        return false;
    }

    map_.print_summary();

    // Step 2: mmap the GGUF file for CPU-side expert weight access
    printf("[Pipeline] Step 2/5: Memory-mapping GGUF file...\n");
#ifndef _WIN32
    mmap_fd_ = open(gguf_path.c_str(), O_RDONLY);
    if (mmap_fd_ < 0) {
        fprintf(stderr, "[Pipeline] Failed to open for mmap: %s\n", gguf_path.c_str());
        return false;
    }

    struct stat st;
    fstat(mmap_fd_, &st);
    mmap_size_ = static_cast<size_t>(st.st_size);

    mmap_base_ = static_cast<uint8_t*>(
        mmap(nullptr, mmap_size_, PROT_READ, MAP_PRIVATE | MAP_POPULATE, mmap_fd_, 0));

    if (mmap_base_ == MAP_FAILED) {
        fprintf(stderr, "[Pipeline] mmap failed for %zu bytes\n", mmap_size_);
        close(mmap_fd_);
        mmap_fd_ = -1;
        mmap_base_ = nullptr;
        return false;
    }

    // Advise kernel: we'll read sequentially per layer, with random expert access
    madvise(mmap_base_, mmap_size_, MADV_RANDOM);

    printf("[Pipeline]   mmap'd %.1f GB at %p\n",
           mmap_size_ / (1024.0 * 1024.0 * 1024.0), (void*)mmap_base_);
#else
    fprintf(stderr, "[Pipeline] Windows mmap not yet implemented\n");
    return false;
#endif

    // Step 3: Initialize expert cache
    printf("[Pipeline] Step 3/5: Allocating expert cache...\n");
    const auto& arch = map_.architecture();

    CacheConfig ccfg;
    ccfg.total_vram_bytes    = config.expert_cache_vram_bytes;
    ccfg.expert_slot_bytes   = arch.expert_weight_bytes / 3;  // Per projection slot
    ccfg.recency_weight      = config.recency_weight;
    ccfg.reserved_per_layer  = config.reserved_hot_per_layer;
    ccfg.n_layers            = arch.n_layers;
    ccfg.n_experts_per_layer = arch.n_experts;

    if (ccfg.expert_slot_bytes == 0) {
        // Fallback: estimate from architecture
        // 3 projections × embed_dim × ffn_dim × ~0.26 bytes/param (IQ2_XXS)
        ccfg.expert_slot_bytes = (arch.embed_dim * arch.expert_ffn_dim * 66) / 256;
    }

    if (!cache_.init(ccfg)) {
        fprintf(stderr, "[Pipeline] Failed to initialize expert cache\n");
        release();
        return false;
    }

    // Step 4: Initialize prefetcher
    printf("[Pipeline] Step 4/5: Initializing prefetcher...\n");
    PrefetchConfig pcfg;
    pcfg.enable_prefetch      = config.enable_prefetch;
    pcfg.enable_layer_ahead   = config.enable_prefetch;
    pcfg.speculative_top_k    = config.speculative_top_k;
    pcfg.enable_coalescing    = config.enable_coalescing;
    pcfg.staging_buffer_bytes = config.staging_buffer_bytes;

    if (!prefetcher_.init(pcfg, &cache_, &map_, mmap_base_)) {
        fprintf(stderr, "[Pipeline] Failed to initialize prefetcher\n");
        release();
        return false;
    }

    // Step 5: Create compute streams
    printf("[Pipeline] Step 5/5: Creating GPU streams...\n");
    auto err = GPU_STREAM_CREATE(&compute_stream_);
    if (err != GPU_SUCCESS) {
        fprintf(stderr, "[Pipeline] Failed to create compute stream\n");
        release();
        return false;
    }

    err = GPU_STREAM_CREATE(&expert_stream_);
    if (err != GPU_SUCCESS) {
        fprintf(stderr, "[Pipeline] Failed to create expert stream\n");
        release();
        return false;
    }

    err = GPU_EVENT_CREATE(&expert_ready_event_);
    if (err != GPU_SUCCESS) {
        fprintf(stderr, "[Pipeline] Failed to create expert event\n");
        release();
        return false;
    }

    ready_ = true;
    printf("[Pipeline] ✓ Initialized successfully\n");
    printf("[Pipeline]   Model: %s (%u layers, %u experts, top-%u)\n",
           arch.model_name.c_str(), arch.n_layers, arch.n_experts, arch.n_experts_used);
    printf("[Pipeline]   Cache: %zu slots (%.1f MB), Prefetch: %s\n",
           cache_.n_slots(),
           cache_.n_slots() * ccfg.expert_slot_bytes / (1024.0 * 1024.0),
           config.enable_prefetch ? "ON" : "OFF");
    printf("[Pipeline]   Streams: compute=%p, expert=%p, transfer=%p\n",
           (void*)compute_stream_, (void*)expert_stream_,
           (void*)prefetcher_.transfer_stream());

    return true;
}

void PipelineController::release() {
    ready_ = false;

    prefetcher_.release();
    cache_.release();

    if (compute_stream_) {
        GPU_STREAM_DESTROY(compute_stream_);
        compute_stream_ = nullptr;
    }
    if (expert_stream_) {
        GPU_STREAM_DESTROY(expert_stream_);
        expert_stream_ = nullptr;
    }
    if (expert_ready_event_) {
        GPU_EVENT_DESTROY(expert_ready_event_);
        expert_ready_event_ = nullptr;
    }

#ifndef _WIN32
    if (mmap_base_ && mmap_base_ != MAP_FAILED) {
        munmap(mmap_base_, mmap_size_);
        mmap_base_ = nullptr;
    }
    if (mmap_fd_ >= 0) {
        close(mmap_fd_);
        mmap_fd_ = -1;
    }
#endif

    mmap_size_ = 0;
    profiles_.clear();
}

void PipelineController::begin_token() {
    if (!ready_) return;

    current_profile_ = {};
    current_profile_.token_id = cache_.token_counter();

    if (config_.enable_profiling) {
        // Will be filled in by process_layer and end_token
    }
}

PipelineController::ExpertPointers PipelineController::process_layer(
    const LayerRouting& routing) {

    ExpertPointers result;
    if (!ready_) return result;

    auto t_start = std::chrono::high_resolution_clock::now();

    const uint32_t layer = routing.layer_id;
    const auto& ids = routing.expert_ids;
    const auto& weights = routing.gate_weights;
    const uint32_t n_active = static_cast<uint32_t>(ids.size());

    result.gate_ptrs.resize(n_active);
    result.up_ptrs.resize(n_active);
    result.down_ptrs.resize(n_active);
    result.weights = weights;

    // 1. Batch cache lookup for all 3 projections of all active experts
    auto lookups = cache_.batch_lookup(layer, ids);

    // 2. Count misses — these need H2D transfer
    uint32_t n_misses = 0;
    std::vector<TransferRequest> miss_requests;

    for (size_t i = 0; i < lookups.size(); ++i) {
        if (lookups[i].result == CacheResult::kMiss) {
            n_misses++;

            // Build transfer request
            uint32_t expert_idx = static_cast<uint32_t>(i / 3);
            uint32_t proj_idx   = static_cast<uint32_t>(i % 3);
            auto proj = static_cast<ExpertProj>(proj_idx);

            const ExpertSlice* slice = map_.get_expert(layer, ids[expert_idx], proj);
            if (slice) {
                TransferRequest req;
                req.layer_id   = layer;
                req.expert_id  = ids[expert_idx];
                req.proj       = proj;
                req.src_cpu    = mmap_base_ + slice->offset;
                req.dst_gpu    = lookups[i].gpu_ptr;
                req.size_bytes = slice->size_bytes;
                miss_requests.push_back(req);
            }
        }
    }

    current_profile_.cache_hits += (n_active * 3 - n_misses);
    current_profile_.cache_misses += n_misses;

    // 3. If there are misses, issue synchronous transfer
    //    (the prefetcher should have pre-loaded most of these)
    if (!miss_requests.empty()) {
        auto t_xfer_start = std::chrono::high_resolution_clock::now();

        // Use the prefetcher's transfer stream for consistency
        for (const auto& req : miss_requests) {
#ifdef __HIP_PLATFORM_AMD__
            hipMemcpyAsync(req.dst_gpu, req.src_cpu, req.size_bytes,
                           hipMemcpyHostToDevice, prefetcher_.transfer_stream());
#elif defined(__CUDACC__) || defined(EXPERTFLOW_CUDA)
            cudaMemcpyAsync(req.dst_gpu, req.src_cpu, req.size_bytes,
                            cudaMemcpyHostToDevice, prefetcher_.transfer_stream());
#else
            memcpy(req.dst_gpu, req.src_cpu, req.size_bytes);
#endif
        }

        // Must wait for transfer before expert compute
        GPU_STREAM_SYNC(prefetcher_.transfer_stream());

        auto t_xfer_end = std::chrono::high_resolution_clock::now();
        current_profile_.expert_transfer_ms +=
            std::chrono::duration<double, std::milli>(t_xfer_end - t_xfer_start).count();
    }

    // 4. Extract GPU pointers for each expert's projections
    for (uint32_t i = 0; i < n_active; ++i) {
        result.gate_ptrs[i] = lookups[i * 3 + 0].gpu_ptr;
        result.up_ptrs[i]   = lookups[i * 3 + 1].gpu_ptr;
        result.down_ptrs[i] = lookups[i * 3 + 2].gpu_ptr;
    }

    // 5. Submit prefetch for NEXT layer's predicted experts
    if (config_.enable_prefetch && layer + 1 < map_.architecture().n_layers) {
        prefetcher_.submit_predicted_prefetch(layer, ids);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    current_profile_.expert_compute_ms +=
        std::chrono::duration<double, std::milli>(t_end - t_start).count();

    return result;
}

void PipelineController::end_token() {
    if (!ready_) return;

    // Sync all streams
    GPU_STREAM_SYNC(compute_stream_);
    GPU_STREAM_SYNC(expert_stream_);
    prefetcher_.sync();

    // Advance LRU clock
    cache_.advance_token();

    // Record profiling
    current_profile_.total_ms = current_profile_.attention_ms +
                                 current_profile_.expert_compute_ms +
                                 current_profile_.expert_transfer_ms;

    if (config_.enable_profiling) {
        profiles_.push_back(current_profile_);
    }
}

void PipelineController::warmup_profile(size_t n_warmup_tokens) {
    if (!ready_) return;

    printf("[Pipeline] Warmup profiling with %zu tokens...\n", n_warmup_tokens);

    const auto& arch = map_.architecture();
    std::mt19937 rng(42);  // Fixed seed for reproducibility
    std::uniform_int_distribution<uint32_t> expert_dist(0, arch.n_experts - 1);

    // Track expert frequency across warmup tokens
    // frequency[layer][expert_id] = count
    std::vector<std::vector<uint32_t>> frequency(
        arch.n_layers, std::vector<uint32_t>(arch.n_experts, 0));

    for (size_t tok = 0; tok < n_warmup_tokens; ++tok) {
        begin_token();

        for (uint32_t layer = 0; layer < arch.n_layers; ++layer) {
            // Simulate random routing (real routing would come from the router MLP)
            LayerRouting routing;
            routing.layer_id = layer;
            routing.expert_ids.resize(arch.n_experts_used);
            routing.gate_weights.resize(arch.n_experts_used, 1.0f / arch.n_experts_used);

            for (uint32_t i = 0; i < arch.n_experts_used; ++i) {
                routing.expert_ids[i] = expert_dist(rng);
                frequency[layer][routing.expert_ids[i]]++;
            }

            process_layer(routing);
        }

        end_token();
    }

    // Find top-K hot experts per layer
    hot_experts_.resize(arch.n_layers);
    for (uint32_t layer = 0; layer < arch.n_layers; ++layer) {
        // Sort experts by frequency
        std::vector<uint32_t> indices(arch.n_experts);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                  [&](uint32_t a, uint32_t b) {
                      return frequency[layer][a] > frequency[layer][b];
                  });

        hot_experts_[layer].assign(
            indices.begin(),
            indices.begin() + std::min(static_cast<size_t>(config_.reserved_hot_per_layer),
                                        indices.size()));
    }

    // Reserve hot experts in cache
    cache_.reserve_hot_experts(hot_experts_);

    // Reset stats after warmup
    cache_.reset_stats();
    prefetcher_.reset_stats();

    printf("[Pipeline] Warmup complete. Reserved %u hot experts per layer.\n",
           config_.reserved_hot_per_layer);
}

PipelineStats PipelineController::compute_stats() const {
    PipelineStats s{};

    const auto& cs = cache_.stats();
    const auto& ps = prefetcher_.stats();

    s.tokens_generated = cache_.token_counter();
    s.avg_cache_hit_rate = cs.hit_rate();
    s.avg_prefetch_accuracy = ps.prefetch_efficiency();
    s.total_pipeline_stalls = ps.pipeline_stalls;

    if (!profiles_.empty()) {
        double total_ms = 0, total_attn = 0, total_expert = 0, total_xfer = 0;
        double peak_tps = 0;

        for (const auto& p : profiles_) {
            total_ms += p.total_ms;
            total_attn += p.attention_ms;
            total_expert += p.expert_compute_ms;
            total_xfer += p.expert_transfer_ms;

            double tps = (p.total_ms > 0) ? 1000.0 / p.total_ms : 0;
            peak_tps = std::max(peak_tps, tps);
        }

        s.total_time_ms = total_ms;
        s.avg_tok_per_sec = (total_ms > 0) ? profiles_.size() * 1000.0 / total_ms : 0;
        s.peak_tok_per_sec = peak_tps;

        if (total_ms > 0) {
            s.pct_attention = total_attn * 100.0 / total_ms;
            s.pct_expert_compute = total_expert * 100.0 / total_ms;
            s.pct_expert_transfer = total_xfer * 100.0 / total_ms;
            s.pct_overhead = 100.0 - s.pct_attention - s.pct_expert_compute - s.pct_expert_transfer;
        }
    }

    return s;
}

void PipelineController::print_report() const {
    printf("\n");
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║          ExpertFlow Performance Report           ║\n");
    printf("╠══════════════════════════════════════════════════╣\n");

    const auto& arch = map_.architecture();
    printf("║ Model: %-42s ║\n", arch.model_name.c_str());
    printf("║ Architecture: %-35s ║\n", arch.architecture.c_str());
    printf("║ Layers: %u, Experts: %u, Top-%u                   ║\n",
           arch.n_layers, arch.n_experts, arch.n_experts_used);
    printf("╠══════════════════════════════════════════════════╣\n");

    auto s = compute_stats();
    printf("║ Tokens generated: %-31lu ║\n", s.tokens_generated);
    printf("║ Avg speed: %-22.1f tok/s       ║\n", s.avg_tok_per_sec);
    printf("║ Peak speed: %-21.1f tok/s       ║\n", s.peak_tok_per_sec);
    printf("║ Total time: %-21.1f ms          ║\n", s.total_time_ms);
    printf("╠══════════════════════════════════════════════════╣\n");

    printf("║ Cache hit rate: %-18.1f%%              ║\n", s.avg_cache_hit_rate * 100.0);
    printf("║ Prefetch accuracy: %-15.1f%%              ║\n", s.avg_prefetch_accuracy * 100.0);
    printf("║ Pipeline stalls: %-32lu ║\n", s.total_pipeline_stalls);
    printf("╠══════════════════════════════════════════════════╣\n");

    printf("║ Time breakdown:                                  ║\n");
    printf("║   Attention:      %6.1f%%                        ║\n", s.pct_attention);
    printf("║   Expert compute: %6.1f%%                        ║\n", s.pct_expert_compute);
    printf("║   Expert transfer:%6.1f%%                        ║\n", s.pct_expert_transfer);
    printf("║   Overhead:       %6.1f%%                        ║\n", s.pct_overhead);
    printf("╚══════════════════════════════════════════════════╝\n");

    printf("\n");
    cache_.print_status();
    printf("\n");
    prefetcher_.print_status();
}

}  // namespace expertflow
