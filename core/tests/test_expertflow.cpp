// ExpertFlow — Unit tests for expert_map and expert_cache
// Tests against any MoE GGUF model (set EXPERTFLOW_MODEL env var)

#include "expertflow/expert_cache.h"
#include "expertflow/expert_compressor.h"
#include "expertflow/expert_map.h"
#include "expertflow/expert_prefetcher.h"
#include "expertflow/expertflow_backend.h"
#include "expertflow/moe_dispatch.h"
#include "expertflow/pipeline_controller.h"
#include "expertflow/routing_predictor.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    static void test_##name(); \
    struct Register_##name { \
        Register_##name() { \
            printf("  TEST %-50s ", #name); \
            try { test_##name(); printf("PASS\n"); tests_passed++; } \
            catch (...) { printf("FAIL (exception)\n"); tests_failed++; } \
        } \
    } register_##name; \
    static void test_##name()

#define ASSERT_TRUE(expr) \
    do { if (!(expr)) { printf("FAIL\n    %s:%d: %s\n", __FILE__, __LINE__, #expr); tests_failed++; return; } } while(0)

#define ASSERT_EQ(a, b) \
    do { if ((a) != (b)) { printf("FAIL\n    %s:%d: %s != %s (%lu vs %lu)\n", __FILE__, __LINE__, #a, #b, (unsigned long)(a), (unsigned long)(b)); tests_failed++; return; } } while(0)

#define ASSERT_GT(a, b) \
    do { if (!((a) > (b))) { printf("FAIL\n    %s:%d: %s > %s (%lu vs %lu)\n", __FILE__, __LINE__, #a, #b, (unsigned long)(a), (unsigned long)(b)); tests_failed++; return; } } while(0)

#define ASSERT_NEAR(a, b, tol) \
    do { if (std::fabs((a) - (b)) > (tol)) { printf("FAIL\n    %s:%d: %s ≈ %s (%.4f vs %.4f, tol=%.4f)\n", __FILE__, __LINE__, #a, #b, (double)(a), (double)(b), (double)(tol)); tests_failed++; return; } } while(0)

// Path to any MoE model (set via EXPERTFLOW_MODEL env var)
static std::string get_model_path() {
    const char* env = getenv("EXPERTFLOW_MODEL");
    if (env) return env;
    return "../../models/Qwen3.5-122B-A10B-UD-IQ2_XXS.gguf";
}

// ============================================================
// ExpertMap tests
// ============================================================

TEST(expert_map_load_valid) {
    expertflow::ExpertMap map;
    bool ok = map.load(get_model_path());
    ASSERT_TRUE(ok);
    ASSERT_TRUE(map.is_moe());
}

TEST(expert_map_architecture) {
    expertflow::ExpertMap map;
    map.load(get_model_path());
    const auto& arch = map.architecture();

    // Must have valid MoE metadata (works for any model)
    ASSERT_TRUE(arch.n_layers > 0);
    ASSERT_TRUE(arch.n_experts > 1);  // MoE needs >1 expert
    ASSERT_TRUE(arch.n_experts_used > 0);
    ASSERT_TRUE(arch.n_experts_used <= arch.n_experts);
    ASSERT_TRUE(arch.embed_dim > 0);
    ASSERT_TRUE(arch.expert_ffn_dim > 0);
    ASSERT_TRUE(arch.n_heads > 0);
    ASSERT_TRUE(arch.n_kv_heads > 0);
    ASSERT_TRUE(!arch.architecture.empty());
    ASSERT_TRUE(!arch.model_name.empty());
    printf("\n      %s: %s, %u layers, %u experts, top-%u, dim=%u, ffn=%u",
           arch.model_name.c_str(), arch.architecture.c_str(),
           arch.n_layers, arch.n_experts, arch.n_experts_used,
           arch.embed_dim, arch.expert_ffn_dim);
}

TEST(expert_map_tensor_counts) {
    expertflow::ExpertMap map;
    map.load(get_model_path());

    // Should have expert slices: 48 layers × 256 experts × 3 projections = 36,864
    // (or fewer if gate_up is fused: 48 × 256 × 3 still since we split them)
    ASSERT_GT(map.expert_slices().size(), 0u);

    // Should have shared tensors (attention, norms, embeddings, router)
    ASSERT_GT(map.shared_tensors().size(), 0u);

    printf("\n      shared=%zu, expert_slices=%zu",
           map.shared_tensors().size(), map.expert_slices().size());
}

TEST(expert_map_expert_lookup) {
    expertflow::ExpertMap map;
    map.load(get_model_path());
    const auto& arch = map.architecture();

    // Look up first expert in first layer
    const auto* slice = map.get_expert(0, 0, expertflow::ExpertProj::kGateProj);
    ASSERT_TRUE(slice != nullptr);
    ASSERT_EQ(slice->layer_id, 0u);
    ASSERT_EQ(slice->expert_id, 0u);
    ASSERT_GT(slice->size_bytes, 0u);

    // Look up last expert in last layer (dynamic from model)
    uint32_t last_layer = arch.n_layers - 1;
    uint32_t last_expert = arch.n_experts - 1;
    const auto* last = map.get_expert(last_layer, last_expert,
                                       expertflow::ExpertProj::kDownProj);
    ASSERT_TRUE(last != nullptr);
    ASSERT_EQ(last->layer_id, last_layer);
    ASSERT_EQ(last->expert_id, last_expert);

    // Non-existent expert should return nullptr
    const auto* bad = map.get_expert(arch.n_layers + 1, 0,
                                      expertflow::ExpertProj::kGateProj);
    ASSERT_TRUE(bad == nullptr);
}

TEST(expert_map_active_experts) {
    expertflow::ExpertMap map;
    map.load(get_model_path());
    const auto& arch = map.architecture();

    // Generate top-K expert IDs within model's valid range
    uint32_t top_k = arch.n_experts_used;
    std::vector<uint32_t> ids(top_k);
    for (uint32_t i = 0; i < top_k; ++i) {
        ids[i] = (i * 31) % arch.n_experts;  // Spread across expert range
    }
    auto active = map.get_active_experts(0, ids);

    // Should return top_k experts × 3 projections
    ASSERT_EQ(active.size(), static_cast<size_t>(top_k * 3u));

    // Each slice should belong to layer 0
    for (const auto* s : active) {
        ASSERT_EQ(s->layer_id, 0u);
    }
}

TEST(expert_map_expert_size) {
    expertflow::ExpertMap map;
    map.load(get_model_path());
    const auto& arch = map.architecture();

    // Per-expert size must be positive
    double expert_mb = arch.expert_weight_bytes / (1024.0 * 1024.0);
    printf("\n      expert_weight=%.2f MB", expert_mb);
    ASSERT_GT(expert_mb, 0.01);    // At least some bytes
    ASSERT_TRUE(expert_mb < 100.0); // Sanity upper bound

    // Total expert pool: n_layers × n_experts × per_expert
    double total_gb = arch.total_expert_bytes / (1024.0 * 1024.0 * 1024.0);
    printf(", total_experts=%.1f GB", total_gb);
    ASSERT_GT(total_gb, 0.01);     // Must have some expert weight

    // Consistency: total ≈ n_layers × n_experts × per_expert
    uint64_t expected = static_cast<uint64_t>(arch.n_layers) *
                        arch.n_experts * arch.expert_weight_bytes;
    ASSERT_EQ(arch.total_expert_bytes, expected);
}

TEST(expert_map_speed_estimates) {
    expertflow::ExpertMap map;
    map.load(get_model_path());
    const auto& arch = map.architecture();

    // Speed estimates must be positive and finite
    double gpu_speed = arch.estimate_speed(288e9);
    printf("\n      gpu=%.0f tok/s", gpu_speed);
    ASSERT_GT(gpu_speed, 0.0);

    double ddr4_speed = arch.estimate_speed(40e9);
    printf(", ddr4=%.0f tok/s", ddr4_speed);
    ASSERT_GT(ddr4_speed, 0.0);

    // Higher bandwidth → higher speed
    ASSERT_TRUE(gpu_speed > ddr4_speed);
}

TEST(expert_map_print_summary) {
    expertflow::ExpertMap map;
    map.load(get_model_path());
    printf("\n");
    map.print_summary();
}

// ============================================================
// ExpertCache tests (CPU fallback mode — no GPU needed)
// ============================================================

TEST(cache_init_and_release) {
    expertflow::CacheConfig cfg;
    cfg.total_vram_bytes    = 100 * 1024 * 1024;  // 100 MB for testing
    cfg.expert_slot_bytes   = 1024 * 1024;         // 1 MB per slot
    cfg.recency_weight      = 0.6f;
    cfg.reserved_per_layer  = 0;
    cfg.n_layers            = 48;
    cfg.n_experts_per_layer = 256;

    expertflow::ExpertCache cache;
    bool ok = cache.init(cfg);
    ASSERT_TRUE(ok);
    ASSERT_EQ(cache.n_slots(), 100u);
    ASSERT_EQ(cache.n_occupied(), 0u);

    cache.release();
    ASSERT_EQ(cache.n_slots(), 0u);
}

TEST(cache_hit_miss) {
    expertflow::CacheConfig cfg;
    cfg.total_vram_bytes    = 10 * 1024 * 1024;  // 10 MB
    cfg.expert_slot_bytes   = 1024 * 1024;        // 1 MB per slot → 10 slots
    cfg.recency_weight      = 0.6f;
    cfg.reserved_per_layer  = 0;
    cfg.n_layers            = 48;
    cfg.n_experts_per_layer = 256;

    expertflow::ExpertCache cache;
    cache.init(cfg);

    // First lookup should be a MISS
    auto r1 = cache.lookup(0, 10, expertflow::ExpertProj::kGateProj);
    ASSERT_TRUE(r1.result == expertflow::CacheResult::kMiss);
    ASSERT_TRUE(r1.gpu_ptr != nullptr);

    // Second lookup for same expert should be a HIT
    auto r2 = cache.lookup(0, 10, expertflow::ExpertProj::kGateProj);
    ASSERT_TRUE(r2.result == expertflow::CacheResult::kHit);
    ASSERT_TRUE(r2.gpu_ptr == r1.gpu_ptr);  // Same GPU pointer

    ASSERT_EQ(cache.stats().hits, 1u);
    ASSERT_EQ(cache.stats().misses, 1u);
}

TEST(cache_eviction_lru) {
    expertflow::CacheConfig cfg;
    cfg.total_vram_bytes    = 5 * 1024 * 1024;   // 5 slots
    cfg.expert_slot_bytes   = 1024 * 1024;
    cfg.recency_weight      = 1.0f;  // Pure LRU
    cfg.reserved_per_layer  = 0;
    cfg.n_layers            = 48;
    cfg.n_experts_per_layer = 256;

    expertflow::ExpertCache cache;
    cache.init(cfg);

    // Fill all 5 slots
    for (uint32_t i = 0; i < 5; ++i) {
        cache.lookup(0, i, expertflow::ExpertProj::kGateProj);
        cache.advance_token();
    }
    ASSERT_EQ(cache.n_occupied(), 5u);

    // Inserting a 6th should evict expert 0 (oldest, pure LRU)
    cache.lookup(0, 99, expertflow::ExpertProj::kGateProj);
    ASSERT_EQ(cache.n_occupied(), 5u);  // Still 5 slots

    // Expert 4 (most recent) should still be cached
    auto r4 = cache.lookup(0, 4, expertflow::ExpertProj::kGateProj);
    ASSERT_TRUE(r4.result == expertflow::CacheResult::kHit);

    // Expert 99 (just inserted) should also be cached
    auto r99 = cache.lookup(0, 99, expertflow::ExpertProj::kGateProj);
    ASSERT_TRUE(r99.result == expertflow::CacheResult::kHit);

    // Total evictions should be exactly 1 (expert 0 was evicted)
    ASSERT_EQ(cache.stats().evictions, 1u);
}

TEST(cache_frequency_weighted) {
    expertflow::CacheConfig cfg;
    cfg.total_vram_bytes    = 3 * 1024 * 1024;   // 3 slots
    cfg.expert_slot_bytes   = 1024 * 1024;
    cfg.recency_weight      = 0.0f;  // Pure frequency
    cfg.reserved_per_layer  = 0;
    cfg.n_layers            = 48;
    cfg.n_experts_per_layer = 256;

    expertflow::ExpertCache cache;
    cache.init(cfg);

    // Fill 3 slots: expert 0 accessed 10 times, expert 1 once, expert 2 once
    cache.lookup(0, 0, expertflow::ExpertProj::kGateProj);
    for (int i = 0; i < 9; ++i) {
        cache.advance_token();
        cache.lookup(0, 0, expertflow::ExpertProj::kGateProj);  // 10 hits
    }
    cache.advance_token();
    cache.lookup(0, 1, expertflow::ExpertProj::kGateProj);  // 1 access
    cache.advance_token();
    cache.lookup(0, 2, expertflow::ExpertProj::kGateProj);  // 1 access

    // Insert 4th expert — should evict expert 1 or 2 (lowest frequency), NOT expert 0
    cache.advance_token();
    cache.lookup(0, 99, expertflow::ExpertProj::kGateProj);

    // Expert 0 should still be cached (high frequency)
    auto r = cache.lookup(0, 0, expertflow::ExpertProj::kGateProj);
    ASSERT_TRUE(r.result == expertflow::CacheResult::kHit);
}

TEST(cache_batch_lookup) {
    expertflow::CacheConfig cfg;
    cfg.total_vram_bytes    = 100 * 1024 * 1024;  // 100 MB
    cfg.expert_slot_bytes   = 1024 * 1024;
    cfg.recency_weight      = 0.6f;
    cfg.reserved_per_layer  = 0;
    cfg.n_layers            = 48;
    cfg.n_experts_per_layer = 256;

    expertflow::ExpertCache cache;
    cache.init(cfg);

    // Batch lookup 8 experts (like Qwen3.5-122B top-8)
    std::vector<uint32_t> ids = {10, 42, 100, 200, 3, 77, 250, 15};
    auto results = cache.batch_lookup(5, ids);

    // Should return 8 experts × 3 projections = 24 results
    ASSERT_EQ(results.size(), 24u);

    // All should be misses (first time)
    for (const auto& r : results) {
        ASSERT_TRUE(r.result == expertflow::CacheResult::kMiss);
    }

    // Second batch lookup should be all hits
    auto results2 = cache.batch_lookup(5, ids);
    for (const auto& r : results2) {
        ASSERT_TRUE(r.result == expertflow::CacheResult::kHit);
    }

    ASSERT_NEAR(cache.stats().hit_rate(), 0.5, 0.01);  // 24 hits / 48 total
}

TEST(cache_hit_rate_report) {
    expertflow::CacheConfig cfg;
    cfg.total_vram_bytes    = 100 * 1024 * 1024;
    cfg.expert_slot_bytes   = 1024 * 1024;
    cfg.recency_weight      = 0.6f;
    cfg.reserved_per_layer  = 0;
    cfg.n_layers            = 48;
    cfg.n_experts_per_layer = 256;

    expertflow::ExpertCache cache;
    cache.init(cfg);

    // Simulate 100 tokens, each accessing 8 experts per layer (layer 0 only)
    // Use a mix of repeat and new experts to simulate realistic routing
    uint32_t hot_experts[] = {10, 42, 100, 200, 3};  // 5 hot
    for (int tok = 0; tok < 100; ++tok) {
        for (int i = 0; i < 5; ++i) {
            cache.lookup(0, hot_experts[i], expertflow::ExpertProj::kGateProj);
        }
        // 3 random experts per token
        cache.lookup(0, tok % 256, expertflow::ExpertProj::kGateProj);
        cache.lookup(0, (tok * 7) % 256, expertflow::ExpertProj::kGateProj);
        cache.lookup(0, (tok * 13) % 256, expertflow::ExpertProj::kGateProj);
        cache.advance_token();
    }

    printf("\n      ");
    cache.print_status();

    // Hit rate should be well above 50% due to hot expert reuse
    ASSERT_GT(cache.stats().hit_rate(), 0.5);
}

// ============================================================
// ExpertPrefetcher tests (CPU fallback mode)
// ============================================================

TEST(prefetcher_init_and_release) {
    // Need a map and cache to init the prefetcher
    expertflow::ExpertMap map;
    map.load(get_model_path());

    expertflow::CacheConfig ccfg;
    ccfg.total_vram_bytes    = 50 * 1024 * 1024;  // 50 MB
    ccfg.expert_slot_bytes   = map.architecture().expert_weight_bytes / 3;
    ccfg.recency_weight      = 0.6f;
    ccfg.reserved_per_layer  = 0;
    ccfg.n_layers            = 48;
    ccfg.n_experts_per_layer = 256;

    expertflow::ExpertCache cache;
    cache.init(ccfg);

    // mmap the model file for CPU-side access
    int fd = open(get_model_path().c_str(), O_RDONLY);
    ASSERT_TRUE(fd >= 0);
    struct stat st;
    fstat(fd, &st);
    auto* base = static_cast<uint8_t*>(
        mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0));
    ASSERT_TRUE(base != MAP_FAILED);

    expertflow::PrefetchConfig pcfg;
    pcfg.enable_prefetch      = true;
    pcfg.enable_layer_ahead   = true;
    pcfg.speculative_top_k    = 12;
    pcfg.enable_coalescing    = true;
    pcfg.staging_buffer_bytes = 32 * 1024 * 1024;  // 32 MB

    expertflow::ExpertPrefetcher prefetcher;
    bool ok = prefetcher.init(pcfg, &cache, &map, base);
    ASSERT_TRUE(ok);

    prefetcher.release();
    munmap(base, st.st_size);
    close(fd);
}

TEST(prefetcher_submit_and_await) {
    expertflow::ExpertMap map;
    map.load(get_model_path());

    expertflow::CacheConfig ccfg;
    ccfg.total_vram_bytes    = 100 * 1024 * 1024;  // 100 MB
    ccfg.expert_slot_bytes   = map.architecture().expert_weight_bytes / 3;
    ccfg.recency_weight      = 0.6f;
    ccfg.reserved_per_layer  = 0;
    ccfg.n_layers            = 48;
    ccfg.n_experts_per_layer = 256;

    expertflow::ExpertCache cache;
    cache.init(ccfg);

    int fd = open(get_model_path().c_str(), O_RDONLY);
    struct stat st;
    fstat(fd, &st);
    auto* base = static_cast<uint8_t*>(
        mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0));

    expertflow::PrefetchConfig pcfg;
    pcfg.enable_prefetch      = true;
    pcfg.enable_layer_ahead   = true;
    pcfg.speculative_top_k    = 12;
    pcfg.enable_coalescing    = true;
    pcfg.staging_buffer_bytes = 32 * 1024 * 1024;

    expertflow::ExpertPrefetcher prefetcher;
    prefetcher.init(pcfg, &cache, &map, base);

    // Submit prefetch for 8 experts at layer 0
    std::vector<uint32_t> ids = {10, 42, 100, 200, 3, 77, 250, 15};
    prefetcher.submit_prefetch(0, ids);
    bool ok = prefetcher.await_prefetch();
    ASSERT_TRUE(ok);

    // After prefetch, all those experts should be in cache (hits)
    for (uint32_t eid : ids) {
        auto r = cache.lookup(0, eid, expertflow::ExpertProj::kGateProj);
        ASSERT_TRUE(r.result == expertflow::CacheResult::kHit);
    }

    // Submit predicted prefetch for layer 1 based on layer 0's routing
    prefetcher.submit_predicted_prefetch(0, ids);
    ok = prefetcher.await_prefetch();
    ASSERT_TRUE(ok);

    printf("\n      ");
    prefetcher.print_status();

    prefetcher.release();
    munmap(base, st.st_size);
    close(fd);
}

TEST(prefetcher_data_integrity) {
    // Verify that prefetched data matches the original GGUF data
    expertflow::ExpertMap map;
    map.load(get_model_path());

    expertflow::CacheConfig ccfg;
    ccfg.total_vram_bytes    = 50 * 1024 * 1024;
    ccfg.expert_slot_bytes   = map.architecture().expert_weight_bytes / 3;
    ccfg.recency_weight      = 0.6f;
    ccfg.reserved_per_layer  = 0;
    ccfg.n_layers            = 48;
    ccfg.n_experts_per_layer = 256;

    expertflow::ExpertCache cache;
    cache.init(ccfg);

    int fd = open(get_model_path().c_str(), O_RDONLY);
    struct stat st;
    fstat(fd, &st);
    auto* base = static_cast<uint8_t*>(
        mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0));

    expertflow::PrefetchConfig pcfg;
    pcfg.enable_prefetch      = true;
    pcfg.enable_layer_ahead   = false;
    pcfg.speculative_top_k    = 0;
    pcfg.enable_coalescing    = true;
    pcfg.staging_buffer_bytes = 32 * 1024 * 1024;

    expertflow::ExpertPrefetcher prefetcher;
    prefetcher.init(pcfg, &cache, &map, base);

    // Prefetch expert (5, 42, gate_proj)
    prefetcher.submit_prefetch(5, {42});
    prefetcher.await_prefetch();

    // Get the cache slot GPU pointer
    auto lookup = cache.lookup(5, 42, expertflow::ExpertProj::kGateProj);
    ASSERT_TRUE(lookup.result == expertflow::CacheResult::kHit);

    // Get the original data from mmap
    const auto* slice = map.get_expert(5, 42, expertflow::ExpertProj::kGateProj);
    ASSERT_TRUE(slice != nullptr);
    const uint8_t* original = base + slice->offset;

    // In CPU fallback mode, the "GPU" pointer is just malloc'd memory
    // The prefetcher should have copied the data there
    int cmp = memcmp(lookup.gpu_ptr, original, slice->size_bytes);
    ASSERT_EQ(cmp, 0);

    printf("\n      Verified %lu bytes of expert (5, 42, gate) data integrity",
           slice->size_bytes);

    prefetcher.release();
    munmap(base, st.st_size);
    close(fd);
}

// ============================================================
// PipelineController integration test
// ============================================================

TEST(pipeline_init_and_process) {
    expertflow::PipelineConfig cfg = expertflow::PipelineConfig::defaults();
    cfg.expert_cache_vram_bytes = 100 * 1024 * 1024;  // 100 MB for testing
    cfg.enable_profiling = true;
    cfg.enable_prefetch = true;

    expertflow::PipelineController pipeline;
    bool ok = pipeline.init(get_model_path(), cfg);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(pipeline.is_ready());

    // Simulate generating 5 tokens
    for (int tok = 0; tok < 5; ++tok) {
        pipeline.begin_token();

        // Process 3 layers (not all 48 — just to test the flow)
        for (uint32_t layer = 0; layer < 3; ++layer) {
            expertflow::LayerRouting routing;
            routing.layer_id = layer;
            routing.expert_ids = {10, 42, 100, 200, 3, 77, 250, 15};
            routing.gate_weights = {0.15f, 0.14f, 0.13f, 0.12f,
                                    0.12f, 0.12f, 0.11f, 0.11f};

            auto ptrs = pipeline.process_layer(routing);

            // Should get 8 expert pointers for each projection
            ASSERT_EQ(ptrs.gate_ptrs.size(), 8u);
            ASSERT_EQ(ptrs.up_ptrs.size(), 8u);
            ASSERT_EQ(ptrs.down_ptrs.size(), 8u);
            ASSERT_EQ(ptrs.weights.size(), 8u);

            // All pointers should be non-null
            for (int i = 0; i < 8; ++i) {
                ASSERT_TRUE(ptrs.gate_ptrs[i] != nullptr);
                ASSERT_TRUE(ptrs.up_ptrs[i] != nullptr);
                ASSERT_TRUE(ptrs.down_ptrs[i] != nullptr);
            }
        }

        pipeline.end_token();
    }

    // Check cache hit rate improves over tokens (same experts reused)
    double hit_rate = pipeline.cache().stats().hit_rate();
    printf("\n      tokens=5, hit_rate=%.1f%%", hit_rate * 100.0);
    ASSERT_GT(hit_rate, 0.3);  // Should have some hits from repeated routing

    // Print full report
    printf("\n");
    pipeline.print_report();

    pipeline.release();
    ASSERT_TRUE(!pipeline.is_ready());
}

TEST(pipeline_cache_warmup_simulation) {
    // Simulate realistic MoE routing with locality.
    // Hot working set: 48 layers × 20 hot experts = 960 (layer, expert) pairs.
    // Cache must be large enough to hold the hot set: 960 × 0.84 MB ≈ 806 MB.
    expertflow::CacheConfig ccfg;
    ccfg.total_vram_bytes    = 1024ULL * 1024 * 1024;  // 1 GB
    ccfg.expert_slot_bytes   = 860 * 1024;              // ~0.84 MB per projection
    ccfg.recency_weight      = 0.6f;
    ccfg.reserved_per_layer  = 0;
    ccfg.n_layers            = 48;
    ccfg.n_experts_per_layer = 256;

    expertflow::ExpertCache cache;
    cache.init(ccfg);

    // Simulate 200 tokens with realistic routing:
    // - 62.5% of activations go to 20 "hot" experts (5/8)
    // - 37.5% go to random experts (3/8)
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> hot_dist(0, 19);
    std::uniform_int_distribution<uint32_t> cold_dist(20, 255);

    for (int tok = 0; tok < 200; ++tok) {
        for (uint32_t layer = 0; layer < 48; ++layer) {
            // 8 experts per layer: 5 hot + 3 cold
            for (int i = 0; i < 5; ++i) {
                uint32_t eid = hot_dist(rng);
                cache.lookup(layer, eid, expertflow::ExpertProj::kGateProj);
            }
            for (int i = 0; i < 3; ++i) {
                uint32_t eid = cold_dist(rng);
                cache.lookup(layer, eid, expertflow::ExpertProj::kGateProj);
            }
        }
        cache.advance_token();
    }

    double hit_rate = cache.stats().hit_rate();
    printf("\n      200 tokens × 48 layers × 8 experts: hit_rate=%.1f%%", hit_rate * 100.0);
    printf("\n      ");
    cache.print_status();

    // With 1GB cache holding full hot set, expect >50% hit rate
    ASSERT_GT(hit_rate, 0.5);
}

// ============================================================
// MoE Dispatch tests (CPU reference)
// ============================================================

TEST(dispatch_fp32_identity) {
    // Test MoE dispatch with identity-like FP32 weights (no quantization)
    // 1 token, embed=4, ffn=2, 1 expert with weight=1.0
    const uint32_t B = 1, E = 4, F = 2, K = 1;

    // Input: [1, 0, 0, 0]
    float input[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float output[4] = {};
    size_t scratch_sz = expertflow::moe_scratch_bytes(B, F);
    std::vector<float> scratch(scratch_sz / sizeof(float));

    // Identity-ish weights (F32, type=0)
    // gate_proj [F×E] = [[1,0,0,0],[0,1,0,0]]  (2×4 matrix, row-major)
    float w_gate[8] = {1,0,0,0, 0,1,0,0};
    float w_up[8]   = {1,0,0,0, 0,1,0,0};
    // down_proj [E×F] = [[1,0],[0,1],[0,0],[0,0]]  (4×2 matrix)
    float w_down[8] = {1,0, 0,1, 0,0, 0,0};

    const uint8_t* gp = reinterpret_cast<const uint8_t*>(w_gate);
    const uint8_t* up = reinterpret_cast<const uint8_t*>(w_up);
    const uint8_t* dp = reinterpret_cast<const uint8_t*>(w_down);
    float gw = 1.0f;

    expertflow::MoeDispatchParams params{};
    params.input = input;
    params.batch_size = B;
    params.embed_dim = E;
    params.ffn_dim = F;
    params.n_active = K;
    params.gate_weights = &gw;
    params.gate_ptrs = &gp;
    params.up_ptrs = &up;
    params.down_ptrs = &dp;
    params.quant_type = 0;  // F32
    params.shared_gate_ptr = nullptr;
    params.shared_up_ptr = nullptr;
    params.shared_down_ptr = nullptr;
    params.output = output;
    params.scratch = scratch.data();
    params.scratch_bytes = scratch_sz;

    auto stats = expertflow::moe_dispatch_cpu(params);

    // gate_proj(input) = [1, 0], up_proj(input) = [1, 0]
    // SiLU([1,0]) = [0.7311, 0.0]
    // hidden = SiLU(gate) * up = [0.7311, 0.0]
    // down_proj(hidden) = [0.7311, 0.0, 0.0, 0.0]
    // output = 1.0 * [0.7311, 0, 0, 0]
    float expected_silu = 1.0f / (1.0f + std::exp(-1.0f));  // σ(1) ≈ 0.7311
    ASSERT_NEAR(output[0], expected_silu, 0.001f);
    ASSERT_NEAR(output[1], 0.0f, 0.001f);
    ASSERT_NEAR(output[2], 0.0f, 0.001f);
    ASSERT_NEAR(output[3], 0.0f, 0.001f);

    printf("\n      output=[%.4f, %.4f, %.4f, %.4f], %.2f GFLOP/s, %.3f ms",
           output[0], output[1], output[2], output[3],
           stats.gflops, stats.total_ms);
}

TEST(dispatch_multi_expert_accumulation) {
    // Test that multiple experts are weighted and accumulated correctly
    const uint32_t B = 1, E = 2, F = 2, K = 2;

    float input[2] = {1.0f, 1.0f};
    float output[2] = {};
    size_t scratch_sz = expertflow::moe_scratch_bytes(B, F);
    std::vector<float> scratch(scratch_sz / sizeof(float));

    // Expert 0: identity-like
    float w_gate0[4] = {1,0, 0,1};
    float w_up0[4]   = {1,0, 0,1};
    float w_down0[4] = {1,0, 0,1};

    // Expert 1: scaled
    float w_gate1[4] = {2,0, 0,2};
    float w_up1[4]   = {1,0, 0,1};
    float w_down1[4] = {1,0, 0,1};

    const uint8_t* gp[2] = {reinterpret_cast<const uint8_t*>(w_gate0),
                             reinterpret_cast<const uint8_t*>(w_gate1)};
    const uint8_t* up[2] = {reinterpret_cast<const uint8_t*>(w_up0),
                             reinterpret_cast<const uint8_t*>(w_up1)};
    const uint8_t* dp[2] = {reinterpret_cast<const uint8_t*>(w_down0),
                             reinterpret_cast<const uint8_t*>(w_down1)};
    float gw[2] = {0.6f, 0.4f};

    expertflow::MoeDispatchParams params{};
    params.input = input;
    params.batch_size = B;
    params.embed_dim = E;
    params.ffn_dim = F;
    params.n_active = K;
    params.gate_weights = gw;
    params.gate_ptrs = gp;
    params.up_ptrs = up;
    params.down_ptrs = dp;
    params.quant_type = 0;
    params.shared_gate_ptr = nullptr;
    params.shared_up_ptr = nullptr;
    params.shared_down_ptr = nullptr;
    params.output = output;
    params.scratch = scratch.data();
    params.scratch_bytes = scratch_sz;

    auto stats = expertflow::moe_dispatch_cpu(params);

    // Expert 0: gate=[1,1], up=[1,1], SiLU([1,1])=[0.731,0.731], hidden=[0.731,0.731]
    //           down(hidden)=[0.731, 0.731]
    // Expert 1: gate=[2,2], up=[1,1], SiLU([2,2])=[1.762,1.762], hidden=[1.762,1.762]
    //           down(hidden)=[1.762, 1.762]
    // output = 0.6*[0.731,0.731] + 0.4*[1.762,1.762]
    float silu1 = 1.0f / (1.0f + std::exp(-1.0f));  // 0.7311
    float silu2 = 2.0f / (1.0f + std::exp(-2.0f));  // 1.7616
    float expected = 0.6f * silu1 + 0.4f * silu2;

    ASSERT_NEAR(output[0], expected, 0.01f);
    ASSERT_NEAR(output[1], expected, 0.01f);

    printf("\n      output=[%.4f, %.4f], expected=%.4f, %.3f ms",
           output[0], output[1], expected, stats.total_ms);
}

TEST(dispatch_flops_calculation) {
    // Verify FLOP counting for Qwen3.5-122B-A10B dimensions
    uint64_t flops = expertflow::moe_flops(1, 3072, 1024, 8);
    // Per expert: 2*(3072*1024) + 2*(1024*3072) = 2*3145728 + 2*3145728 = 12582912
    // × 8 experts = 100663296 ≈ 100.7 MFLOP
    ASSERT_EQ(flops, 8ULL * (2ULL * 3072 * 1024 + 2ULL * 1024 * 3072));
    printf("\n      1 token MoE FLOPs: %.1f MFLOP", flops / 1e6);
}

TEST(dispatch_silu_correctness) {
    // Verify SiLU implementation
    float vals[5] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    expertflow::silu_f32(vals, 5);

    // SiLU(x) = x * σ(x) = x / (1 + exp(-x))
    auto silu_ref = [](float x) { return x / (1.0f + std::exp(-x)); };

    ASSERT_NEAR(vals[0], silu_ref(-2.0f), 0.0001f);
    ASSERT_NEAR(vals[1], silu_ref(-1.0f), 0.0001f);
    ASSERT_NEAR(vals[2], 0.0f, 0.0001f);
    ASSERT_NEAR(vals[3], silu_ref(1.0f), 0.0001f);
    ASSERT_NEAR(vals[4], silu_ref(2.0f), 0.0001f);
}

// ============================================================
// Backend integration tests
// ============================================================

TEST(backend_tensor_classification) {
    // Pattern A: stacked expert tensors (Qwen, DeepSeek, Llama4) → CPU
    ASSERT_TRUE(expertflow::is_expert_tensor("blk.5.ffn_gate_exps.weight"));
    ASSERT_TRUE(expertflow::is_expert_tensor("blk.10.ffn_up_exps.weight"));
    ASSERT_TRUE(expertflow::is_expert_tensor("blk.47.ffn_down_exps.weight"));
    ASSERT_TRUE(expertflow::is_expert_tensor("blk.0.ffn_gate_up_exps.weight"));

    // Pattern B: per-expert tensors (Mixtral 8x7B/8x22B) → CPU
    ASSERT_TRUE(expertflow::is_expert_tensor("blk.0.ffn_gate.0.weight"));
    ASSERT_TRUE(expertflow::is_expert_tensor("blk.5.ffn_up.3.weight"));
    ASSERT_TRUE(expertflow::is_expert_tensor("blk.31.ffn_down.7.weight"));
    ASSERT_TRUE(expertflow::is_expert_tensor("blk.10.ffn_gate.15.weight"));

    // Shared tensors → GPU (must NOT match expert patterns)
    ASSERT_TRUE(expertflow::is_shared_tensor("blk.5.attn_q.weight"));
    ASSERT_TRUE(expertflow::is_shared_tensor("blk.5.attn_k.weight"));
    ASSERT_TRUE(expertflow::is_shared_tensor("blk.5.ffn_norm.weight"));
    ASSERT_TRUE(expertflow::is_shared_tensor("blk.5.ffn_gate_inp.weight"));
    ASSERT_TRUE(expertflow::is_shared_tensor("token_embd.weight"));
    ASSERT_TRUE(expertflow::is_shared_tensor("output.weight"));
    ASSERT_TRUE(expertflow::is_shared_tensor("blk.21.ssm_out.weight"));
    // Shared FFN (no _exps suffix, no .{E} suffix)
    ASSERT_TRUE(expertflow::is_shared_tensor("blk.5.ffn_gate.weight"));
    ASSERT_TRUE(expertflow::is_shared_tensor("blk.5.ffn_up.weight"));
    ASSERT_TRUE(expertflow::is_shared_tensor("blk.5.ffn_down.weight"));

    // Placement classification
    ASSERT_TRUE(expertflow::classify_tensor("blk.5.ffn_gate_exps.weight") ==
                expertflow::TensorPlacement::kCPU);
    ASSERT_TRUE(expertflow::classify_tensor("blk.0.ffn_gate.0.weight") ==
                expertflow::TensorPlacement::kCPU);
    ASSERT_TRUE(expertflow::classify_tensor("blk.5.attn_q.weight") ==
                expertflow::TensorPlacement::kGPU);
}

TEST(backend_create_and_prepare) {
    expertflow::BackendConfig cfg = expertflow::BackendConfig::defaults();
    cfg.expert_cache_bytes = 100 * 1024 * 1024;  // 100 MB for testing
    cfg.enable_profiling = true;

    auto backend = expertflow::ExpertFlowBackend::create(get_model_path(), cfg);
    ASSERT_TRUE(backend != nullptr);
    ASSERT_TRUE(backend->is_active());

    const auto& arch = backend->architecture();
    ASSERT_TRUE(arch.n_layers > 0);
    ASSERT_TRUE(arch.n_experts > 0);
    ASSERT_TRUE(arch.n_experts_used > 0);
    ASSERT_TRUE(arch.embed_dim > 0);
    printf("\n      model=%s, layers=%u, experts=%u, top-%u",
           arch.model_name.c_str(), arch.n_layers, arch.n_experts,
           arch.n_experts_used);

    // Simulate 3 tokens using model's actual top-K
    uint32_t top_k = arch.n_experts_used;
    for (int tok = 0; tok < 3; ++tok) {
        backend->begin_token();

        for (uint32_t layer = 0; layer < 3; ++layer) {
            // Generate top-K expert IDs within valid range
            std::vector<uint32_t> ids(top_k);
            std::vector<float> weights(top_k);
            for (uint32_t i = 0; i < top_k; ++i) {
                ids[i] = (layer * 13 + tok * 7 + i * 31) % arch.n_experts;
                weights[i] = 1.0f / static_cast<float>(top_k);
            }

            auto ptrs = backend->prepare_experts(layer, ids, weights);
            ASSERT_EQ(ptrs.gate_ptrs.size(), static_cast<size_t>(top_k));

            for (uint32_t i = 0; i < top_k; ++i) {
                ASSERT_TRUE(ptrs.gate_ptrs[i] != nullptr);
            }
        }

        backend->end_token();
    }

    auto s = backend->stats();
    printf("\n      tokens=3, cache_hit=%.1f%%, prefetch_acc=%.1f%%",
           s.avg_cache_hit_rate * 100.0, s.avg_prefetch_accuracy * 100.0);
}

TEST(backend_global_instance) {
    // Test global backend management
    ASSERT_TRUE(!expertflow::is_expertflow_active());

    expertflow::BackendConfig cfg = expertflow::BackendConfig::defaults();
    cfg.expert_cache_bytes = 50 * 1024 * 1024;

    auto backend = expertflow::ExpertFlowBackend::create(get_model_path(), cfg);
    ASSERT_TRUE(backend != nullptr);

    expertflow::set_global_backend(std::move(backend));
    ASSERT_TRUE(expertflow::is_expertflow_active());

    auto* g = expertflow::get_global_backend();
    ASSERT_TRUE(g != nullptr);
    ASSERT_TRUE(g->architecture().n_layers > 0);

    // Reset
    expertflow::set_global_backend(nullptr);
    ASSERT_TRUE(!expertflow::is_expertflow_active());
}

// ============================================================
// Phase A: KV Compression → VRAM Budget Tests
// ============================================================

TEST(auto_config_kv_compression_ratio) {
    expertflow::ExpertMap map;
    map.load(get_model_path());
    const auto& arch = map.architecture();

    size_t vram = 6ULL * 1024 * 1024 * 1024;  // 6 GB

    // Without TurboQuant KV (ratio=1.0) — 200 MB KV overhead
    auto cfg_no_tq = expertflow::PipelineConfig::auto_config(vram, arch, 1.0f);
    // With TQ3 (ratio=3.5/16 ≈ 0.22) — ~44 MB KV overhead
    auto cfg_tq3 = expertflow::PipelineConfig::auto_config(vram, arch, 3.5f / 16.0f);
    // With TQ2 (ratio=2.5/16 ≈ 0.16) — ~32 MB KV overhead
    auto cfg_tq2 = expertflow::PipelineConfig::auto_config(vram, arch, 2.5f / 16.0f);

    // TQ3 should give MORE expert cache than no compression
    ASSERT_TRUE(cfg_tq3.expert_cache_vram_bytes > cfg_no_tq.expert_cache_vram_bytes);
    // TQ2 should give even MORE than TQ3
    ASSERT_TRUE(cfg_tq2.expert_cache_vram_bytes > cfg_tq3.expert_cache_vram_bytes);

    // Verify the savings magnitude
    size_t tq3_savings = cfg_tq3.expert_cache_vram_bytes -
                         cfg_no_tq.expert_cache_vram_bytes;
    // TQ3 saves ~156 MB of KV VRAM (200 - 44 = 156)
    ASSERT_TRUE(tq3_savings > 100ULL * 1024 * 1024);  // At least 100 MB
    ASSERT_TRUE(tq3_savings < 200ULL * 1024 * 1024);  // At most 200 MB

    printf("\n      no_tq=%.0f MB, tq3=%.0f MB, tq2=%.0f MB (savings=%.0f MB)",
           cfg_no_tq.expert_cache_vram_bytes / (1024.0 * 1024.0),
           cfg_tq3.expert_cache_vram_bytes / (1024.0 * 1024.0),
           cfg_tq2.expert_cache_vram_bytes / (1024.0 * 1024.0),
           tq3_savings / (1024.0 * 1024.0));
}

TEST(auto_config_backend_kv_compression) {
    expertflow::ExpertMap map;
    map.load(get_model_path());
    const auto& arch = map.architecture();

    size_t vram = 6ULL * 1024 * 1024 * 1024;

    auto cfg_no_tq = expertflow::BackendConfig::auto_config(vram, arch, 1.0f);
    auto cfg_tq3   = expertflow::BackendConfig::auto_config(vram, arch, 3.5f / 16.0f);

    // Backend auto_config should also benefit from TQ3
    ASSERT_TRUE(cfg_tq3.expert_cache_bytes > cfg_no_tq.expert_cache_bytes);

    // Extra slots from TQ3 savings
    size_t extra_slots = (cfg_tq3.expert_cache_bytes - cfg_no_tq.expert_cache_bytes) /
                         arch.expert_weight_bytes;
    printf("\n      extra_slots=%zu from TQ3 KV savings", extra_slots);
    ASSERT_TRUE(extra_slots > 30);  // Should gain at least 30 extra expert slots
}

// ============================================================
// Phase C: Expert Transfer Compression Tests
// ============================================================

TEST(compress_roundtrip_synthetic) {
    // Create data with repeating patterns (like quantized weights)
    std::vector<uint8_t> original(4096);
    std::mt19937 rng(42);
    // IQ2_XXS-like: 66-byte blocks with repeating codebook indices
    for (size_t i = 0; i < original.size(); i += 66) {
        // FP16 scale (2 bytes, varies)
        original[i] = rng() & 0xFF;
        original[i + 1] = rng() & 0xFF;
        // Codebook indices (64 bytes, many repeats)
        for (size_t j = 2; j < 66 && i + j < original.size(); ++j) {
            original[i + j] = (rng() % 8);  // Low entropy codebook
        }
    }

    auto compressed = expertflow::compress_expert(original.data(), original.size());
    ASSERT_TRUE(!compressed.empty());
    ASSERT_TRUE(compressed.size() < original.size());

    // Decompress
    size_t orig_size = expertflow::compressed_original_size(
        compressed.data(), compressed.size());
    ASSERT_EQ(orig_size, original.size());

    std::vector<uint8_t> decompressed(orig_size);
    size_t dec_size = expertflow::decompress_expert(
        decompressed.data(), compressed.data(), compressed.size());
    ASSERT_EQ(dec_size, original.size());

    // Verify byte-perfect round-trip
    ASSERT_TRUE(memcmp(original.data(), decompressed.data(), original.size()) == 0);

    double ratio = static_cast<double>(compressed.size()) / original.size();
    printf("\n      synthetic: %zu → %zu bytes (%.1f%% ratio)",
           original.size(), compressed.size(), ratio * 100.0);
}

TEST(compress_real_expert_data) {
    // Load a real expert from the model and compress it
    expertflow::ExpertMap map;
    map.load(get_model_path());

    // Open model file and mmap
    int fd = open(get_model_path().c_str(), O_RDONLY);
    ASSERT_TRUE(fd >= 0);
    struct stat st;
    fstat(fd, &st);
    void* base = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    ASSERT_TRUE(base != MAP_FAILED);

    // Get first expert's gate projection
    const auto* slice = map.get_expert(0, 0, expertflow::ExpertProj::kGateProj);
    ASSERT_TRUE(slice != nullptr);
    ASSERT_TRUE(slice->size_bytes > 0);

    const uint8_t* expert_data = static_cast<const uint8_t*>(base) +
                                  slice->offset;

    auto compressed = expertflow::compress_expert(expert_data, slice->size_bytes);

    if (!compressed.empty()) {
        // Verify round-trip on real data
        std::vector<uint8_t> decompressed(slice->size_bytes);
        size_t dec = expertflow::decompress_expert(
            decompressed.data(), compressed.data(), compressed.size());
        ASSERT_EQ(dec, slice->size_bytes);
        ASSERT_TRUE(memcmp(expert_data, decompressed.data(), slice->size_bytes) == 0);

        double ratio = static_cast<double>(compressed.size()) / slice->size_bytes;
        double savings_pct = (1.0 - ratio) * 100.0;
        printf("\n      real expert: %zu → %zu bytes (%.1f%% saved, ratio=%.2f)",
               slice->size_bytes, compressed.size(), savings_pct, ratio);
    } else {
        // Compression not beneficial — still valid
        printf("\n      real expert: %zu bytes (incompressible)", slice->size_bytes);
    }

    munmap(base, st.st_size);
    close(fd);
}

TEST(pinned_memory_pool) {
    expertflow::PinnedMemoryPool pool;
    ASSERT_TRUE(pool.init(1024 * 1024));  // 1 MB pool
    ASSERT_TRUE(pool.is_initialized());
    ASSERT_EQ(pool.capacity(), 1024ULL * 1024);
    ASSERT_EQ(pool.used(), 0ULL);

    // Acquire some regions
    uint8_t* a = pool.acquire(256);
    ASSERT_TRUE(a != nullptr);
    uint8_t* b = pool.acquire(512);
    ASSERT_TRUE(b != nullptr);
    ASSERT_TRUE(b != a);  // Different regions
    ASSERT_TRUE(pool.used() >= 768);  // At least 256 + 512 (may include alignment)

    // Write and verify
    memset(a, 0xAA, 256);
    memset(b, 0xBB, 512);
    ASSERT_EQ(a[0], 0xAA);
    ASSERT_EQ(b[0], 0xBB);

    // Reset and reuse
    pool.reset();
    ASSERT_EQ(pool.used(), 0ULL);
    uint8_t* c = pool.acquire(1024);
    ASSERT_TRUE(c != nullptr);

    // Overflow: try to acquire more than capacity
    pool.reset();
    uint8_t* big = pool.acquire(2 * 1024 * 1024);  // 2 MB > 1 MB pool
    ASSERT_TRUE(big == nullptr);

    pool.release();
    ASSERT_TRUE(!pool.is_initialized());
    printf("\n      pool: 1 MB, acquire/reset/overflow OK");
}

// ============================================================
// Phase D: Adaptive Routing Predictor Tests
// ============================================================

TEST(predictor_init_and_observe) {
    expertflow::AdaptiveRoutingPredictor predictor;
    auto cfg = expertflow::PredictorConfig::defaults(48, 256, 8);
    predictor.init(cfg);
    ASSERT_TRUE(predictor.is_initialized());

    // Observe routing for a few layers
    predictor.begin_token();
    predictor.observe(0, {10, 20, 30, 40, 50, 60, 70, 80});
    predictor.observe(1, {11, 21, 31, 41, 51, 61, 71, 81});
    predictor.observe(2, {10, 20, 30, 40, 50, 60, 70, 80});

    // Predict for layer 1 given layer 0's routing
    auto pred = predictor.predict(0, {10, 20, 30, 40, 50, 60, 70, 80});
    ASSERT_TRUE(!pred.expert_ids.empty());
    ASSERT_TRUE(pred.expert_ids.size() <= cfg.predict_count);
    printf("\n      predicted %zu experts for layer 1", pred.expert_ids.size());
}

TEST(predictor_learns_patterns) {
    expertflow::AdaptiveRoutingPredictor predictor;
    auto cfg = expertflow::PredictorConfig::defaults(4, 32, 4);
    cfg.predict_count = 6;
    predictor.init(cfg);

    // Simulate 50 tokens with a consistent pattern:
    // Layer 0 always uses experts {0,1,2,3}
    // Layer 1 always uses experts {4,5,6,7}
    // Layer 2 always uses experts {0,1,2,3}
    // Layer 3 always uses experts {8,9,10,11}
    std::vector<std::vector<uint32_t>> pattern = {
        {0, 1, 2, 3}, {4, 5, 6, 7}, {0, 1, 2, 3}, {8, 9, 10, 11}
    };

    for (int tok = 0; tok < 50; ++tok) {
        predictor.begin_token();
        for (uint32_t l = 0; l < 4; ++l) {
            predictor.observe(l, pattern[l]);
        }
    }

    // Now predict: given layer 0 routing, predict layer 1
    auto pred1 = predictor.predict(0, pattern[0]);
    // Should strongly predict experts {4,5,6,7}
    uint32_t correct = 0;
    for (uint32_t p : pred1.expert_ids) {
        for (uint32_t a : pattern[1]) {
            if (p == a) { ++correct; break; }
        }
    }
    float accuracy = static_cast<float>(correct) / pattern[1].size();
    printf("\n      pattern accuracy: %.0f%% (%u/%zu correct)",
           accuracy * 100.0f, correct, pattern[1].size());
    ASSERT_TRUE(accuracy >= 0.75f);  // Should predict at least 3/4 correctly

    // Predict layer 3 from layer 2
    auto pred3 = predictor.predict(2, pattern[2]);
    uint32_t correct3 = 0;
    for (uint32_t p : pred3.expert_ids) {
        for (uint32_t a : pattern[3]) {
            if (p == a) { ++correct3; break; }
        }
    }
    float acc3 = static_cast<float>(correct3) / pattern[3].size();
    printf(", layer3_acc=%.0f%%", acc3 * 100.0f);
    ASSERT_TRUE(acc3 >= 0.75f);
}

TEST(predictor_accuracy_improves) {
    expertflow::AdaptiveRoutingPredictor predictor;
    auto cfg = expertflow::PredictorConfig::defaults(4, 64, 4);
    cfg.predict_count = 6;
    predictor.init(cfg);

    // Simulate with a semi-random but biased pattern
    std::mt19937 rng(123);

    // Hot experts: layers tend to use nearby experts
    auto gen_routing = [&](uint32_t layer) -> std::vector<uint32_t> {
        uint32_t base = layer * 8;  // Each layer favors a different range
        std::vector<uint32_t> ids(4);
        for (int i = 0; i < 4; ++i) {
            ids[i] = (base + i + (rng() % 4)) % 64;
        }
        return ids;
    };

    // Phase 1: first 10 tokens — predictor is cold
    uint32_t early_correct = 0, early_total = 0;
    for (int tok = 0; tok < 10; ++tok) {
        predictor.begin_token();
        std::vector<uint32_t> prev_routing;
        for (uint32_t l = 0; l < 4; ++l) {
            auto actual = gen_routing(l);
            if (l > 0 && !prev_routing.empty()) {
                auto pred = predictor.predict(l - 1, prev_routing);
                for (uint32_t a : actual) {
                    for (uint32_t p : pred.expert_ids) {
                        if (p == a) { ++early_correct; break; }
                    }
                }
                early_total += actual.size();
            }
            predictor.observe(l, actual);
            prev_routing = actual;
        }
    }

    // Phase 2: next 50 tokens — predictor has learned
    uint32_t late_correct = 0, late_total = 0;
    for (int tok = 0; tok < 50; ++tok) {
        predictor.begin_token();
        std::vector<uint32_t> prev_routing;
        for (uint32_t l = 0; l < 4; ++l) {
            auto actual = gen_routing(l);
            if (l > 0 && !prev_routing.empty()) {
                auto pred = predictor.predict(l - 1, prev_routing);
                for (uint32_t a : actual) {
                    for (uint32_t p : pred.expert_ids) {
                        if (p == a) { ++late_correct; break; }
                    }
                }
                late_total += actual.size();
            }
            predictor.observe(l, actual);
            prev_routing = actual;
        }
    }

    float early_acc = early_total > 0
        ? static_cast<float>(early_correct) / early_total : 0.0f;
    float late_acc = late_total > 0
        ? static_cast<float>(late_correct) / late_total : 0.0f;

    printf("\n      early_acc=%.0f%%, late_acc=%.0f%% (should improve)",
           early_acc * 100.0f, late_acc * 100.0f);

    // Late accuracy should be better than early (predictor learns)
    ASSERT_TRUE(late_acc >= early_acc);
}

// ============================================================
// Phase B: GPU Kernel Path Tests
// ============================================================

TEST(gpu_dispatch_fallback) {
    // Without GPU hardware, moe_dispatch_gpu should fall back to CPU
    expertflow::MoeDispatchParams params{};
    uint32_t E = 64, F = 32, K = 2;

    std::vector<float> input(E, 1.0f);
    std::vector<float> output(E, 0.0f);
    std::vector<float> scratch(3 * F);
    std::vector<float> gate_weights = {0.6f, 0.4f};

    // Create fake FP32 weight matrices
    std::vector<float> w1(F * E, 0.01f);
    std::vector<float> w2(F * E, 0.02f);
    std::vector<float> w3(E * F, 0.01f);
    std::vector<float> w4(E * F, 0.02f);
    std::vector<float> w5(F * E, 0.01f);
    std::vector<float> w6(F * E, 0.02f);

    const uint8_t* gate_ptrs[2] = {
        reinterpret_cast<const uint8_t*>(w1.data()),
        reinterpret_cast<const uint8_t*>(w2.data())};
    const uint8_t* up_ptrs[2] = {
        reinterpret_cast<const uint8_t*>(w3.data()),
        reinterpret_cast<const uint8_t*>(w4.data())};
    const uint8_t* down_ptrs[2] = {
        reinterpret_cast<const uint8_t*>(w5.data()),
        reinterpret_cast<const uint8_t*>(w6.data())};

    params.input = input.data();
    params.batch_size = 1;
    params.embed_dim = E;
    params.ffn_dim = F;
    params.n_active = K;
    params.gate_weights = gate_weights.data();
    params.gate_ptrs = gate_ptrs;
    params.up_ptrs = up_ptrs;
    params.down_ptrs = down_ptrs;
    params.quant_type = 0;  // F32
    params.output = output.data();
    params.scratch = scratch.data();
    params.scratch_bytes = scratch.size() * sizeof(float);

    // GPU dispatch with nullptr stream should use CPU path
    auto stats = expertflow::moe_dispatch_gpu(params, nullptr);
    ASSERT_TRUE(stats.total_ms >= 0.0);
    ASSERT_TRUE(stats.flops > 0);

    printf("\n      cpu_fallback: %.3f ms, %.1f GFLOP/s", stats.total_ms, stats.gflops);
}

// ============================================================
// Main
// ============================================================

int main() {
    printf("=== ExpertFlow Tests ===\n");
    printf("\n");

    // Force construction of all test registrations
    // (they run in static init order, which is fine for a simple test runner)

    printf("\n=== Results: %d passed, %d failed ===\n",
           tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
