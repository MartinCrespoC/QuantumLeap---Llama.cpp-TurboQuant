// ExpertFlow — MoE-Aware Inference Engine
// expert_cache.h: GPU-side LRU expert cache with frequency-weighted eviction
//
// Manages a fixed-size pool of GPU memory slots for expert weights.
// Experts are loaded from CPU RAM on demand and cached on GPU.
// Eviction policy: LRU with frequency boost — experts that fire often
// stay cached even if they haven't been used in the last few tokens.
//
// Target: scales to any MoE model — slot count auto-computed from
// VRAM budget ÷ per-expert size (e.g. ~2.5 MB for IQ2_XXS models)

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "expertflow/expert_map.h"

namespace expertflow {

// Result of a cache lookup
enum class CacheResult : uint8_t {
    kHit  = 0,  // Expert is in GPU cache, pointer is valid
    kMiss = 1,  // Expert is NOT in cache, needs H2D transfer
};

// A single slot in the GPU expert cache
struct CacheSlot {
    uint32_t layer_id;     // Which layer this expert belongs to
    uint32_t expert_id;    // Which expert within the layer
    ExpertProj proj;       // Which projection (gate/up/down)
    bool     occupied;     // Is this slot in use?
    uint64_t last_access;  // Token counter at last access (for LRU)
    uint32_t access_count; // Total accesses since load (for frequency)
    uint8_t* gpu_ptr;      // Pointer to GPU memory for this slot
};

// Lookup result returned by the cache
struct CacheLookup {
    CacheResult result;
    uint8_t*    gpu_ptr;    // GPU pointer (valid if HIT, target buffer if MISS)
    size_t      slot_index; // Index into the slot array
};

// Statistics for monitoring cache performance
struct CacheStats {
    uint64_t total_lookups;
    uint64_t hits;
    uint64_t misses;
    uint64_t evictions;
    uint64_t total_bytes_transferred; // H2D bytes for misses

    double hit_rate() const {
        return total_lookups > 0
            ? static_cast<double>(hits) / static_cast<double>(total_lookups)
            : 0.0;
    }

    double miss_rate() const { return 1.0 - hit_rate(); }
};

// Configuration for the expert cache
struct CacheConfig {
    size_t   total_vram_bytes;      // Total VRAM budget for expert cache
    size_t   expert_slot_bytes;     // Bytes per expert slot (from ExpertMap)
    float    recency_weight;        // Weight for recency in eviction score [0,1]
                                    // 1.0 = pure LRU, 0.0 = pure frequency
    uint32_t reserved_per_layer;    // Hot experts per layer reserved from eviction
    uint32_t n_layers;              // Number of transformer layers
    uint32_t n_experts_per_layer;   // Experts per layer (for validation)

    // Compute number of cache slots
    size_t n_slots() const {
        return expert_slot_bytes > 0 ? total_vram_bytes / expert_slot_bytes : 0;
    }
};

// GPU Expert Cache — manages a fixed pool of VRAM for caching expert weights
//
// Usage:
//   1. Create with CacheConfig
//   2. Call init() to allocate GPU memory
//   3. For each token:
//      a. Call lookup(layer, expert, proj) for each active expert
//      b. On HIT: use returned gpu_ptr directly
//      c. On MISS: copy expert data to returned gpu_ptr (async H2D)
//      d. Call mark_accessed(slot_index) after use
//   4. Call advance_token() after each token to update LRU clock
//
class ExpertCache {
public:
    ExpertCache() = default;
    ~ExpertCache();

    // Non-copyable, movable
    ExpertCache(const ExpertCache&) = delete;
    ExpertCache& operator=(const ExpertCache&) = delete;
    ExpertCache(ExpertCache&&) noexcept;
    ExpertCache& operator=(ExpertCache&&) noexcept;

    // Initialize cache with given configuration.
    // Allocates GPU memory pool.
    // Returns false on allocation failure.
    bool init(const CacheConfig& config);

    // Release all GPU memory
    void release();

    // Look up an expert in the cache.
    // Returns HIT with valid gpu_ptr, or MISS with a gpu_ptr to an evicted slot.
    // On MISS, caller must copy expert data to the returned gpu_ptr.
    CacheLookup lookup(uint32_t layer_id, uint32_t expert_id, ExpertProj proj);

    // Mark a slot as accessed (call after using the expert for compute)
    void mark_accessed(size_t slot_index);

    // Advance the token counter (call once per generated token)
    void advance_token();

    // Reserve hot experts that should never be evicted.
    // Call after profiling to lock frequently-used experts in cache.
    // reserved[layer_id] = list of expert_ids to pin for that layer
    void reserve_hot_experts(
        const std::vector<std::vector<uint32_t>>& reserved);

    // Prefetch hint: inform the cache that these experts will be needed soon.
    // The cache can preemptively evict cold slots to make room.
    void prefetch_hint(uint32_t layer_id,
                       const std::vector<uint32_t>& expert_ids);

    // Batch lookup: look up all 3 projections for a set of experts at one layer.
    // Returns a vector of CacheLookup (one per expert × projection).
    // Order: [expert0_gate, expert0_up, expert0_down, expert1_gate, ...]
    std::vector<CacheLookup> batch_lookup(
        uint32_t layer_id,
        const std::vector<uint32_t>& expert_ids);

    // Get cache statistics
    const CacheStats& stats() const { return stats_; }

    // Reset statistics
    void reset_stats();

    // Current token counter
    uint64_t token_counter() const { return token_counter_; }

    // Number of cache slots
    size_t n_slots() const { return slots_.size(); }

    // Number of occupied slots
    size_t n_occupied() const;

    // Configuration
    const CacheConfig& config() const { return config_; }

    // Print cache status summary
    void print_status() const;

private:
    // Find the slot for a given expert, or SIZE_MAX if not cached
    size_t find_slot(uint32_t layer_id, uint32_t expert_id,
                     ExpertProj proj) const;

    // Find the best slot to evict (lowest score, not reserved)
    size_t find_eviction_candidate() const;

    // Compute eviction score for a slot (higher = more worth keeping)
    double eviction_score(const CacheSlot& slot) const;

    // Is this slot reserved (pinned, not evictable)?
    bool is_reserved(size_t slot_index) const;

    CacheConfig config_{};
    CacheStats  stats_{};
    uint64_t    token_counter_ = 0;

    // GPU memory pool (one contiguous allocation, sliced into slots)
    uint8_t* gpu_pool_ = nullptr;
    size_t   gpu_pool_bytes_ = 0;

    // Slot metadata (CPU-side tracking)
    std::vector<CacheSlot> slots_;

    // Reserved (pinned) slot indices
    std::vector<bool> reserved_mask_;

    // Fast lookup: encode (layer, expert, proj) → slot index
    // Uses the same ExpertKey/ExpertKeyHash from expert_map.h
    std::unordered_map<ExpertKey, size_t, ExpertKeyHash> slot_map_;
};

}  // namespace expertflow
