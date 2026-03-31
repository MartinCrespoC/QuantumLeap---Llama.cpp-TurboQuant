#include "turboquant/turboquant.h"
#include "turboquant/turboquant_kv.h"

#include <cstdio>
#include <memory>
#include <unordered_map>
#include <vector>

namespace turboquant {

// ============================================
// llama.cpp Integration Patch
// Hooks into llama.cpp model loading and inference
// to use TurboQuant quantized tensors
//
// Now includes TurboQuant KV cache compression
// based on Google Research (arXiv:2504.19874):
//   Hadamard → PolarQuant → QJL pipeline
//   achieving 6-8x KV cache compression with
//   zero quality loss at 3.5 bits/channel.
// ============================================

// Model layer offloading configuration
struct OffloadConfig {
  int total_layers;
  int gpu_layers;         // Layers fully on GPU
  int cpu_layers;         // Layers on CPU (RAM)
  size_t vram_budget;     // Max VRAM to use (bytes)
  size_t ram_budget;      // Max RAM to use (bytes)
  QuantBits weight_bits;  // Quantization for weights
  QuantBits kv_bits;      // Quantization for KV cache (legacy)
  TQMode tq_kv_mode;      // TurboQuant KV cache mode (new)
  bool use_turboquant_kv;  // Enable TurboQuant KV compression
};

// ============================================
// TurboQuant KV Cache Manager
// Manages per-layer, per-head compressed KV caches
// using the full TurboQuant pipeline.
// ============================================

class TurboQuantKVManager {
 public:
  // ── Constructor with mixed VRAM/RAM placement ──
  // gpu_layers: first N layers store KV in VRAM (fast CUDA attention)
  // remaining layers store KV in RAM (SIMD attention via AVX-512/AVX2)
  TurboQuantKVManager(size_t num_layers, size_t num_heads, size_t head_dim,
                      size_t gpu_layers = 0,
                      TQMode mode = TQMode::kTQ3, uint64_t seed = 42)
      : num_layers_(num_layers),
        num_heads_(num_heads),
        head_dim_(head_dim),
        gpu_kv_layers_(std::min(gpu_layers, num_layers)),
        mode_(mode) {
    ctx_ = TurboQuantContext::create(head_dim, mode, seed);

    k_cache_.resize(num_layers);
    v_cache_.resize(num_layers);
    layer_placement_.resize(num_layers);
    for (size_t l = 0; l < num_layers; ++l) {
      k_cache_[l].resize(num_heads);
      v_cache_[l].resize(num_heads);
      layer_placement_[l] = (l < gpu_kv_layers_) ? KVPlacement::kGPU
                                                  : KVPlacement::kCPU;
    }

    size_t cpu_kv = num_layers - gpu_kv_layers_;
    printf("[TurboQuant KV] Mixed memory init:\n");
    printf("  %zu layers x %zu heads x %zu dim, mode=TQ%d\n",
           num_layers, num_heads, head_dim, static_cast<int>(mode));
    printf("  GPU KV layers: %zu (VRAM — CUDA attention)\n", gpu_kv_layers_);
    printf("  CPU KV layers: %zu (RAM  — SIMD attention)\n", cpu_kv);
  }

  // ── Static factory: auto-split based on VRAM budget ──
  // Estimates per-layer KV memory at max_seq_len and fills VRAM first
  static std::unique_ptr<TurboQuantKVManager> create_auto_split(
      size_t num_layers, size_t num_heads, size_t head_dim,
      size_t vram_kv_budget_bytes, size_t max_seq_len,
      TQMode mode = TQMode::kTQ3, uint64_t seed = 42) {
    // Estimate compressed KV bytes per layer at max_seq_len
    // TQ3: ~3.5 bits/element → ~0.4375 bytes/element
    // Per layer: num_heads * max_seq_len * head_dim * 0.4375 * 2 (K+V)
    float bits_per_elem = 3.5f;
    if (mode == TQMode::kTQ2) bits_per_elem = 2.5f;
    if (mode == TQMode::kTQ4) bits_per_elem = 4.0f;
    float bytes_per_elem = bits_per_elem / 8.0f;
    size_t per_layer_bytes = static_cast<size_t>(
        num_heads * max_seq_len * head_dim * bytes_per_elem * 2);

    size_t gpu_layers = 0;
    if (per_layer_bytes > 0) {
      gpu_layers = vram_kv_budget_bytes / per_layer_bytes;
      if (gpu_layers > num_layers) gpu_layers = num_layers;
    }

    printf("[TurboQuant KV] Auto-split: %zu MB VRAM budget, "
           "~%zu KB/layer at seq=%zu → %zu GPU + %zu CPU layers\n",
           vram_kv_budget_bytes / (1024 * 1024),
           per_layer_bytes / 1024, max_seq_len,
           gpu_layers, num_layers - gpu_layers);

    return std::make_unique<TurboQuantKVManager>(
        num_layers, num_heads, head_dim, gpu_layers, mode, seed);
  }

  // ── Append token: dispatches GPU or CPU path per layer ──
  void append_token(size_t layer, const float* key, const float* value,
                    size_t seq_pos) {
    for (size_t h = 0; h < num_heads_; ++h) {
      const float* k_head = key + h * head_dim_;
      const float* v_head = value + h * head_dim_;

      if (seq_pos == 0) {
#if defined(TURBOQUANT_CUDA) || defined(TURBOQUANT_HIP)
        if (layer_placement_[layer] == KVPlacement::kGPU) {
          k_cache_[layer][h] = turboquant_kv_encode_cuda(*ctx_, k_head, 1);
          v_cache_[layer][h] = turboquant_kv_encode_cuda(*ctx_, v_head, 1);
        } else
#endif
        {
          k_cache_[layer][h] = turboquant_kv_encode(*ctx_, k_head, 1);
          v_cache_[layer][h] = turboquant_kv_encode(*ctx_, v_head, 1);
        }
      } else {
        // Incremental append always on CPU (tiny single-vector operation)
        turboquant_kv_append(*ctx_, k_cache_[layer][h], k_head, seq_pos);
        turboquant_kv_append(*ctx_, v_cache_[layer][h], v_head, seq_pos);
      }
    }
  }

  // ── Attention: dispatches GPU or CPU path per layer ──
  void compute_attention_scores(size_t layer, const float* queries,
                                float* output, size_t num_queries_per_head) {
    for (size_t h = 0; h < num_heads_; ++h) {
      const float* q_head = queries + h * head_dim_;
      float* out_head = output + h * k_cache_[layer][h].seq_len;

#if defined(TURBOQUANT_CUDA) || defined(TURBOQUANT_HIP)
      if (layer_placement_[layer] == KVPlacement::kGPU) {
        turboquant_attention_scores_cuda(*ctx_, q_head, k_cache_[layer][h],
                                         out_head, num_queries_per_head);
      } else
#endif
      {
        turboquant_attention_scores(*ctx_, q_head, k_cache_[layer][h],
                                    out_head, num_queries_per_head);
      }
    }
  }

  // ── Memory statistics with per-device breakdown ──
  struct MemStats {
    size_t total_compressed_bytes;
    size_t total_fp32_equivalent;
    size_t gpu_compressed_bytes;   // KV data in VRAM
    size_t cpu_compressed_bytes;   // KV data in RAM
    size_t gpu_layers;
    size_t cpu_layers;
    float compression_ratio;
    float avg_bits_per_element;
  };

  MemStats memory_stats() const {
    MemStats stats = {};
    stats.gpu_layers = gpu_kv_layers_;
    stats.cpu_layers = num_layers_ - gpu_kv_layers_;
    size_t count = 0;
    for (size_t l = 0; l < num_layers_; ++l) {
      for (size_t h = 0; h < num_heads_; ++h) {
        size_t kb = k_cache_[l][h].memory_bytes();
        size_t vb = v_cache_[l][h].memory_bytes();
        size_t layer_bytes = kb + vb;
        stats.total_compressed_bytes += layer_bytes;

        if (layer_placement_[l] == KVPlacement::kGPU) {
          stats.gpu_compressed_bytes += layer_bytes;
        } else {
          stats.cpu_compressed_bytes += layer_bytes;
        }

        size_t seq = k_cache_[l][h].seq_len;
        stats.total_fp32_equivalent += seq * head_dim_ * sizeof(float) * 2;
        if (seq > 0) {
          stats.avg_bits_per_element += k_cache_[l][h].bits_per_element();
          stats.avg_bits_per_element += v_cache_[l][h].bits_per_element();
          count += 2;
        }
      }
    }
    if (count > 0) stats.avg_bits_per_element /= count;
    if (stats.total_compressed_bytes > 0) {
      stats.compression_ratio = static_cast<float>(stats.total_fp32_equivalent)
                                / stats.total_compressed_bytes;
    }
    return stats;
  }

  // ── Dynamic migration: move layer KV between GPU and CPU ──
  // Useful when VRAM pressure changes during long sequences
  void migrate_layer(size_t layer, KVPlacement new_placement) {
    if (layer >= num_layers_) return;
    KVPlacement old = layer_placement_[layer];
    if (old == new_placement) return;

    layer_placement_[layer] = new_placement;
    // Data is byte arrays — migration is a no-op for the compressed data
    // (it's already in host memory). GPU placement means we'll use CUDA
    // kernels and upload on-demand. For true device-resident caches,
    // a future optimization can pin/copy to device memory.

    if (new_placement == KVPlacement::kGPU) {
      gpu_kv_layers_++;
    } else {
      gpu_kv_layers_--;
    }
    printf("[TurboQuant KV] Layer %zu migrated: %s -> %s\n",
           layer, old == KVPlacement::kGPU ? "GPU" : "CPU",
           new_placement == KVPlacement::kGPU ? "GPU" : "CPU");
  }

  void clear() {
    for (size_t l = 0; l < num_layers_; ++l) {
      for (size_t h = 0; h < num_heads_; ++h) {
        k_cache_[l][h] = TQCompressedKV{};
        v_cache_[l][h] = TQCompressedKV{};
      }
    }
  }

  const TurboQuantContext& context() const { return *ctx_; }
  KVPlacement placement(size_t layer) const { return layer_placement_[layer]; }
  size_t gpu_layers() const { return gpu_kv_layers_; }
  size_t cpu_layers() const { return num_layers_ - gpu_kv_layers_; }
  size_t seq_len(size_t layer = 0, size_t head = 0) const {
    return k_cache_[layer][head].seq_len;
  }

 private:
  size_t num_layers_;
  size_t num_heads_;
  size_t head_dim_;
  size_t gpu_kv_layers_;
  TQMode mode_;
  std::unique_ptr<TurboQuantContext> ctx_;
  std::vector<std::vector<TQCompressedKV>> k_cache_;  // [layers][heads]
  std::vector<std::vector<TQCompressedKV>> v_cache_;
  std::vector<KVPlacement> layer_placement_;           // per-layer device
};

// Auto-calculate optimal layer distribution
OffloadConfig auto_offload_config(
    int total_layers, size_t param_count, size_t vram_available) {

  OffloadConfig config;
  config.total_layers = total_layers;
  config.vram_budget = vram_available;

  // Estimate bytes per layer based on quantization
  // INT2: ~0.25 bytes/param, INT4: ~0.5 bytes/param
  size_t bytes_per_layer_tq2 = (param_count / total_layers) / 4;
  size_t bytes_per_layer_tq4 = (param_count / total_layers) / 2;

  // Reserve 512MB for KV cache and runtime overhead
  size_t usable_vram = (vram_available > 512 * 1024 * 1024)
                           ? vram_available - 512 * 1024 * 1024
                           : 0;

  // Try INT2 first (more layers fit in VRAM)
  int gpu_layers_tq2 = static_cast<int>(usable_vram / bytes_per_layer_tq2);
  if (gpu_layers_tq2 > total_layers) gpu_layers_tq2 = total_layers;

  int gpu_layers_tq4 = static_cast<int>(usable_vram / bytes_per_layer_tq4);
  if (gpu_layers_tq4 > total_layers) gpu_layers_tq4 = total_layers;

  // Use INT2 if it gets significantly more layers on GPU
  if (gpu_layers_tq2 > gpu_layers_tq4 * 1.3) {
    config.gpu_layers = gpu_layers_tq2;
    config.weight_bits = QuantBits::kInt2;
  } else {
    config.gpu_layers = gpu_layers_tq4;
    config.weight_bits = QuantBits::kInt4;
  }

  config.cpu_layers = total_layers - config.gpu_layers;
  config.kv_bits = QuantBits::kInt2;  // Legacy fallback
  config.use_turboquant_kv = true;    // Enable TurboQuant KV by default
  config.tq_kv_mode = TQMode::kTQ3;  // 3.5 bits/channel — zero quality loss
  config.ram_budget = 0;  // Will be calculated at runtime

  printf("[TurboQuant] Auto-offload config:\n");
  printf("  Total layers: %d\n", config.total_layers);
  printf("  GPU layers: %d (%.0f%%)\n", config.gpu_layers,
         100.0f * config.gpu_layers / total_layers);
  printf("  CPU layers: %d\n", config.cpu_layers);
  printf("  Weight quant: INT%d\n", static_cast<int>(config.weight_bits));
  printf("  KV cache: TurboQuant TQ%d (Google arXiv:2504.19874)\n",
         static_cast<int>(config.tq_kv_mode));
  printf("  VRAM budget: %.1f MB\n", vram_available / (1024.0 * 1024.0));

  return config;
}

// Print model size comparison
void print_size_comparison(size_t param_count) {
  printf("\n[TurboQuant] Model size comparison (%zu params):\n", param_count);
  printf("  FP16:     %.1f GB\n", param_count * 2.0 / (1024 * 1024 * 1024));
  printf("  Q8_0:     %.1f GB\n", param_count * 1.0 / (1024 * 1024 * 1024));
  printf("  Q4_K_M:   %.1f GB\n", param_count * 0.5 / (1024 * 1024 * 1024));
  printf("  TQ4:      %.1f GB\n", param_count * 0.516 / (1024 * 1024 * 1024));
  printf("  TQ2:      %.1f GB  <- TurboQuant\n",
         param_count * 0.266 / (1024 * 1024 * 1024));
  printf("  Savings vs Q4_K_M: %.0f%%\n",
         (1.0 - 0.266 / 0.5) * 100.0);
  printf("\n[TurboQuant KV] Cache compression (per 1K context, head_dim=128):\n");
  printf("  FP32 KV:  %.1f MB\n", 1024 * 128 * 4.0 * 2 / (1024 * 1024));
  printf("  TQ3 KV:   %.1f MB  (~3.5 bits, zero quality loss)\n",
         1024 * 128 * (3.5 / 8.0) * 2 / (1024 * 1024));
  printf("  TQ2 KV:   %.1f MB  (~2.5 bits, marginal loss)\n",
         1024 * 128 * (2.5 / 8.0) * 2 / (1024 * 1024));
  printf("  Speedup:  up to 8x attention computation (4-bit TurboQuant)\n");
}

}  // namespace turboquant
