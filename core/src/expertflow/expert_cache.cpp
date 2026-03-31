// ExpertFlow — expert_cache.cpp
// GPU-side LRU expert cache with frequency-weighted eviction
//
// Manages a fixed pool of GPU memory slots for caching expert weights.
// Eviction: score = recency_weight × (1/age) + (1-recency_weight) × frequency
// Higher score = more worth keeping. Lowest score gets evicted on miss.

#include "expertflow/expert_cache.h"

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstdio>
#include <cstring>
#include <limits>

// GPU memory allocation (HIP or CUDA or CPU fallback)
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#define GPU_MALLOC(ptr, size)   hipMalloc((ptr), (size))
#define GPU_FREE(ptr)           hipFree((ptr))
#define GPU_SUCCESS             hipSuccess
#elif defined(__CUDACC__) || defined(EXPERTFLOW_CUDA)
#include <cuda_runtime.h>
#define GPU_MALLOC(ptr, size)   cudaMalloc((ptr), (size))
#define GPU_FREE(ptr)           cudaFree((ptr))
#define GPU_SUCCESS             cudaSuccess
#else
// CPU fallback for testing
#include <cstdlib>
#define GPU_MALLOC(ptr, size)   (*(ptr) = static_cast<uint8_t*>(malloc(size)), 0)
#define GPU_FREE(ptr)           free((ptr))
#define GPU_SUCCESS             0
#endif

namespace expertflow {

ExpertCache::~ExpertCache() {
    release();
}

ExpertCache::ExpertCache(ExpertCache&& other) noexcept
    : config_(other.config_)
    , stats_(other.stats_)
    , token_counter_(other.token_counter_)
    , gpu_pool_(other.gpu_pool_)
    , gpu_pool_bytes_(other.gpu_pool_bytes_)
    , slots_(std::move(other.slots_))
    , reserved_mask_(std::move(other.reserved_mask_))
    , slot_map_(std::move(other.slot_map_))
{
    other.gpu_pool_ = nullptr;
    other.gpu_pool_bytes_ = 0;
}

ExpertCache& ExpertCache::operator=(ExpertCache&& other) noexcept {
    if (this != &other) {
        release();
        config_         = other.config_;
        stats_          = other.stats_;
        token_counter_  = other.token_counter_;
        gpu_pool_       = other.gpu_pool_;
        gpu_pool_bytes_ = other.gpu_pool_bytes_;
        slots_          = std::move(other.slots_);
        reserved_mask_  = std::move(other.reserved_mask_);
        slot_map_       = std::move(other.slot_map_);
        other.gpu_pool_ = nullptr;
        other.gpu_pool_bytes_ = 0;
    }
    return *this;
}

bool ExpertCache::init(const CacheConfig& config) {
    release();
    config_ = config;

    size_t n = config.n_slots();
    if (n == 0) {
        fprintf(stderr, "[ExpertCache] Zero slots — check config\n");
        return false;
    }

    // Allocate contiguous GPU memory pool
    gpu_pool_bytes_ = n * config.expert_slot_bytes;
    auto err = GPU_MALLOC(&gpu_pool_, gpu_pool_bytes_);
    if (err != GPU_SUCCESS || gpu_pool_ == nullptr) {
        fprintf(stderr, "[ExpertCache] GPU allocation failed: %zu bytes (%.1f GB)\n",
                gpu_pool_bytes_, gpu_pool_bytes_ / (1024.0 * 1024.0 * 1024.0));
        gpu_pool_ = nullptr;
        gpu_pool_bytes_ = 0;
        return false;
    }

    // Initialize slots
    slots_.resize(n);
    for (size_t i = 0; i < n; ++i) {
        slots_[i].layer_id     = 0;
        slots_[i].expert_id    = 0;
        slots_[i].proj         = ExpertProj::kGateProj;
        slots_[i].occupied     = false;
        slots_[i].last_access  = 0;
        slots_[i].access_count = 0;
        slots_[i].gpu_ptr      = gpu_pool_ + i * config.expert_slot_bytes;
    }

    reserved_mask_.resize(n, false);
    slot_map_.reserve(n);
    stats_ = {};
    token_counter_ = 0;

    printf("[ExpertCache] Initialized: %zu slots × %.2f MB = %.1f GB VRAM\n",
           n, config.expert_slot_bytes / (1024.0 * 1024.0),
           gpu_pool_bytes_ / (1024.0 * 1024.0 * 1024.0));

    return true;
}

void ExpertCache::release() {
    if (gpu_pool_) {
        GPU_FREE(gpu_pool_);
        gpu_pool_ = nullptr;
    }
    gpu_pool_bytes_ = 0;
    slots_.clear();
    reserved_mask_.clear();
    slot_map_.clear();
}

CacheLookup ExpertCache::lookup(uint32_t layer_id, uint32_t expert_id,
                                 ExpertProj proj) {
    stats_.total_lookups++;

    // Check if already cached
    size_t idx = find_slot(layer_id, expert_id, proj);
    if (idx != SIZE_MAX) {
        // HIT
        stats_.hits++;
        slots_[idx].last_access = token_counter_;
        slots_[idx].access_count++;
        return CacheLookup{CacheResult::kHit, slots_[idx].gpu_ptr, idx};
    }

    // MISS — find eviction candidate
    stats_.misses++;
    size_t evict_idx = find_eviction_candidate();

    // Evict the old occupant
    if (slots_[evict_idx].occupied) {
        stats_.evictions++;
        ExpertKey old_key{
            slots_[evict_idx].layer_id,
            slots_[evict_idx].expert_id,
            slots_[evict_idx].proj
        };
        slot_map_.erase(old_key);
    }

    // Assign new occupant
    slots_[evict_idx].layer_id     = layer_id;
    slots_[evict_idx].expert_id    = expert_id;
    slots_[evict_idx].proj         = proj;
    slots_[evict_idx].occupied     = true;
    slots_[evict_idx].last_access  = token_counter_;
    slots_[evict_idx].access_count = 1;

    ExpertKey new_key{layer_id, expert_id, proj};
    slot_map_[new_key] = evict_idx;

    stats_.total_bytes_transferred += config_.expert_slot_bytes;

    return CacheLookup{CacheResult::kMiss, slots_[evict_idx].gpu_ptr, evict_idx};
}

std::vector<CacheLookup> ExpertCache::batch_lookup(
    uint32_t layer_id,
    const std::vector<uint32_t>& expert_ids) {

    std::vector<CacheLookup> results;
    results.reserve(expert_ids.size() * static_cast<uint32_t>(ExpertProj::kCount));

    for (uint32_t eid : expert_ids) {
        for (uint32_t p = 0; p < static_cast<uint32_t>(ExpertProj::kCount); ++p) {
            results.push_back(lookup(layer_id, eid, static_cast<ExpertProj>(p)));
        }
    }
    return results;
}

void ExpertCache::mark_accessed(size_t slot_index) {
    if (slot_index < slots_.size()) {
        slots_[slot_index].last_access = token_counter_;
        slots_[slot_index].access_count++;
    }
}

void ExpertCache::advance_token() {
    token_counter_++;
}

void ExpertCache::reserve_hot_experts(
    const std::vector<std::vector<uint32_t>>& reserved) {

    // Clear existing reservations
    std::fill(reserved_mask_.begin(), reserved_mask_.end(), false);

    // For each layer, pre-load and pin the hot experts
    for (size_t layer = 0; layer < reserved.size() && layer < config_.n_layers; ++layer) {
        for (uint32_t eid : reserved[layer]) {
            for (uint32_t p = 0; p < static_cast<uint32_t>(ExpertProj::kCount); ++p) {
                auto proj = static_cast<ExpertProj>(p);
                // Force into cache via lookup
                CacheLookup result = lookup(static_cast<uint32_t>(layer), eid, proj);
                // Mark as reserved
                if (result.slot_index < reserved_mask_.size()) {
                    reserved_mask_[result.slot_index] = true;
                }
            }
        }
    }

    size_t n_reserved = 0;
    for (bool r : reserved_mask_) if (r) n_reserved++;
    printf("[ExpertCache] Reserved %zu slots (%.1f MB) as hot experts\n",
           n_reserved, n_reserved * config_.expert_slot_bytes / (1024.0 * 1024.0));
}

void ExpertCache::prefetch_hint(uint32_t layer_id,
                                 const std::vector<uint32_t>& expert_ids) {
    // Preemptively evict cold slots to make room for predicted experts.
    // This reduces stalls when the actual lookup happens.
    for (uint32_t eid : expert_ids) {
        for (uint32_t p = 0; p < static_cast<uint32_t>(ExpertProj::kCount); ++p) {
            ExpertKey key{layer_id, eid, static_cast<ExpertProj>(p)};
            if (slot_map_.find(key) == slot_map_.end()) {
                // Not cached — find an eviction candidate now
                // (actual copy will happen in submit_prefetch)
                find_eviction_candidate();
            }
        }
    }
}

size_t ExpertCache::find_slot(uint32_t layer_id, uint32_t expert_id,
                               ExpertProj proj) const {
    ExpertKey key{layer_id, expert_id, proj};
    auto it = slot_map_.find(key);
    return (it != slot_map_.end()) ? it->second : SIZE_MAX;
}

size_t ExpertCache::find_eviction_candidate() const {
    double min_score = std::numeric_limits<double>::max();
    size_t min_idx = 0;

    for (size_t i = 0; i < slots_.size(); ++i) {
        // Never evict reserved slots
        if (is_reserved(i)) continue;

        // Prefer empty slots first
        if (!slots_[i].occupied) return i;

        double score = eviction_score(slots_[i]);
        if (score < min_score) {
            min_score = score;
            min_idx = i;
        }
    }
    return min_idx;
}

double ExpertCache::eviction_score(const CacheSlot& slot) const {
    if (!slot.occupied) return -1.0;

    // Recency: higher is better (more recent)
    uint64_t age = token_counter_ - slot.last_access;
    double recency = (age > 0) ? 1.0 / static_cast<double>(age) : 1000.0;

    // Frequency: higher is better (used more often)
    double frequency = static_cast<double>(slot.access_count);

    // Combined score (higher = keep, lower = evict)
    return config_.recency_weight * recency +
           (1.0 - config_.recency_weight) * frequency;
}

bool ExpertCache::is_reserved(size_t slot_index) const {
    return slot_index < reserved_mask_.size() && reserved_mask_[slot_index];
}

size_t ExpertCache::n_occupied() const {
    size_t count = 0;
    for (const auto& slot : slots_) {
        if (slot.occupied) count++;
    }
    return count;
}

void ExpertCache::reset_stats() {
    stats_ = {};
}

void ExpertCache::print_status() const {
    printf("=== ExpertCache Status ===\n");
    printf("Slots: %zu total, %zu occupied, %zu free\n",
           slots_.size(), n_occupied(), slots_.size() - n_occupied());
    printf("VRAM: %.1f GB allocated\n",
           gpu_pool_bytes_ / (1024.0 * 1024.0 * 1024.0));
    printf("Token counter: %lu\n", token_counter_);
    printf("Lookups: %lu (hits: %lu, misses: %lu)\n",
           stats_.total_lookups, stats_.hits, stats_.misses);
    printf("Hit rate: %.1f%%\n", stats_.hit_rate() * 100.0);
    printf("Evictions: %lu\n", stats_.evictions);
    printf("Bytes transferred: %.1f MB\n",
           stats_.total_bytes_transferred / (1024.0 * 1024.0));
    printf("=== End ExpertCache ===\n");
}

}  // namespace expertflow
