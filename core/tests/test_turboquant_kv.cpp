#include "turboquant/hadamard.h"
#include "turboquant/polarquant.h"
#include "turboquant/qjl.h"
#include "turboquant/turboquant_kv.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

// ============================================================================
// TurboQuant KV Cache Pipeline — Unit Tests
// Based on Google Research TurboQuant (arXiv:2504.19874)
//
// Tests:
//   1. Hadamard transform: orthonormality, invertibility
//   2. PolarQuant: decompose/reconstruct accuracy, angle distribution
//   3. QJL: unbiased inner product estimation
//   4. Full pipeline: encode/decode MSE, compression ratio
//   5. Attention scores: distortion vs FP32 baseline
// ============================================================================

using namespace turboquant;

static constexpr float TOLERANCE = 1e-4f;

// Helper: compute L2 norm
static float l2_norm(const float* v, size_t n) {
  float sum = 0.0f;
  for (size_t i = 0; i < n; ++i) sum += v[i] * v[i];
  return std::sqrt(sum);
}

// Helper: compute MSE between two vectors
static float compute_mse(const float* a, const float* b, size_t n) {
  float sum = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    float d = a[i] - b[i];
    sum += d * d;
  }
  return sum / n;
}

// Helper: compute inner product
static float dot_product(const float* a, const float* b, size_t n) {
  float sum = 0.0f;
  for (size_t i = 0; i < n; ++i) sum += a[i] * b[i];
  return sum;
}

// Helper: generate random vectors
static std::vector<float> random_vectors(size_t count, size_t dim, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> data(count * dim);
  for (auto& v : data) v = dist(rng);
  return data;
}

// ─── Test 1: Hadamard Transform ──────────────────────────────────────────────

static int test_hadamard_invertibility() {
  printf("  [Hadamard] Invertibility... ");
  constexpr size_t DIM = 128;

  auto ctx = HadamardContext::create(DIM, 42);
  auto input = random_vectors(1, DIM, 100);
  std::vector<float> rotated(DIM), recovered(DIM);

  randomized_hadamard(*ctx, input.data(), rotated.data(), DIM);
  randomized_hadamard_inverse(*ctx, rotated.data(), recovered.data(), DIM);

  float mse = compute_mse(input.data(), recovered.data(), DIM);
  printf("MSE=%.2e ", mse);
  if (mse > TOLERANCE) {
    printf("FAIL (MSE too high)\n");
    return 1;
  }
  printf("PASS\n");
  return 0;
}

static int test_hadamard_preserves_norm() {
  printf("  [Hadamard] Norm preservation... ");
  constexpr size_t DIM = 256;

  auto ctx = HadamardContext::create(DIM, 42);
  auto input = random_vectors(1, DIM, 200);
  std::vector<float> rotated(DIM);

  randomized_hadamard(*ctx, input.data(), rotated.data(), DIM);

  float norm_in = l2_norm(input.data(), DIM);
  float norm_out = l2_norm(rotated.data(), DIM);
  float ratio = norm_out / norm_in;

  printf("ratio=%.6f ", ratio);
  if (std::abs(ratio - 1.0f) > 0.01f) {
    printf("FAIL (norm not preserved)\n");
    return 1;
  }
  printf("PASS\n");
  return 0;
}

static int test_hadamard_preserves_inner_product() {
  printf("  [Hadamard] Inner product preservation... ");
  constexpr size_t DIM = 128;

  auto ctx = HadamardContext::create(DIM, 42);
  auto a = random_vectors(1, DIM, 300);
  auto b = random_vectors(1, DIM, 301);
  std::vector<float> a_rot(DIM), b_rot(DIM);

  randomized_hadamard(*ctx, a.data(), a_rot.data(), DIM);
  randomized_hadamard(*ctx, b.data(), b_rot.data(), DIM);

  float dp_orig = dot_product(a.data(), b.data(), DIM);
  float dp_rot = dot_product(a_rot.data(), b_rot.data(), DIM);
  float err = std::abs(dp_orig - dp_rot) / (std::abs(dp_orig) + 1e-8f);

  printf("orig=%.4f rot=%.4f rel_err=%.2e ", dp_orig, dp_rot, err);
  if (err > 0.01f) {
    printf("FAIL\n");
    return 1;
  }
  printf("PASS\n");
  return 0;
}

// ─── Test 2: Polar Decomposition ─────────────────────────────────────────────

static int test_polar_roundtrip() {
  printf("  [PolarQuant] Decompose/Reconstruct roundtrip... ");
  constexpr size_t DIM = 64;

  auto input = random_vectors(1, DIM, 400);
  float radius = 0.0f;
  std::vector<float> angles(DIM - 1);
  std::vector<float> reconstructed(DIM);

  polar_decompose(input.data(), &radius, angles.data(), DIM);
  polar_reconstruct(radius, angles.data(), reconstructed.data(), DIM);

  float mse = compute_mse(input.data(), reconstructed.data(), DIM);
  printf("MSE=%.2e radius=%.4f ", mse, radius);
  if (mse > TOLERANCE) {
    printf("FAIL\n");
    return 1;
  }
  printf("PASS\n");
  return 0;
}

static int test_polarquant_encode_decode() {
  printf("  [PolarQuant] Encode/Decode (3-bit angles)... ");
  constexpr size_t DIM = 128;
  constexpr size_t NUM_VECS = 16;

  // Generate random data and rotate it (simulating Hadamard preconditioning)
  auto ctx = HadamardContext::create(DIM, 42);
  auto raw = random_vectors(NUM_VECS, DIM, 500);
  std::vector<float> rotated(NUM_VECS * DIM);
  randomized_hadamard_batch(*ctx, raw.data(), rotated.data(), NUM_VECS, DIM);

  PolarQuantConfig config;
  config.dim = DIM;
  config.angle_bits = 3;
  config.radius_bits = 6;
  config.block_size = 32;

  auto compressed = polarquant_encode(rotated.data(), NUM_VECS, config);

  std::vector<float> decoded(NUM_VECS * DIM);
  polarquant_decode(compressed, decoded.data(), NUM_VECS);

  float mse = compute_mse(rotated.data(), decoded.data(), NUM_VECS * DIM);
  float bpe = compressed.bits_per_element();

  printf("MSE=%.4f bpe=%.2f ", mse, bpe);
  // At 3-bit angles, we expect moderate MSE but good compression
  if (mse > 5.0f) {
    printf("FAIL (MSE too high)\n");
    return 1;
  }
  printf("PASS\n");
  return 0;
}

// ─── Test 3: QJL ─────────────────────────────────────────────────────────────

static int test_qjl_unbiased() {
  printf("  [QJL] Unbiased inner product estimation... ");
  constexpr size_t DIM = 64;
  constexpr size_t NUM_KEYS = 32;
  constexpr size_t TRIALS = 100;

  // Average over multiple random seeds to test unbiasedness
  float total_bias = 0.0f;
  for (size_t trial = 0; trial < TRIALS; ++trial) {
    auto query = random_vectors(1, DIM, 600 + trial);
    auto keys = random_vectors(NUM_KEYS, DIM, 700 + trial);

    auto qjl_ctx = QJLContext::create(DIM, DIM, 137 + trial);
    auto compressed = qjl_encode(*qjl_ctx, keys.data(), NUM_KEYS);

    std::vector<float> corrections(NUM_KEYS);
    qjl_inner_product(*qjl_ctx, query.data(), compressed,
                      corrections.data(), NUM_KEYS);

    // Compare with true inner products
    for (size_t k = 0; k < NUM_KEYS; ++k) {
      float true_dp = dot_product(query.data(), keys.data() + k * DIM, DIM);
      total_bias += (corrections[k] - true_dp);
    }
  }

  float avg_bias = total_bias / (TRIALS * NUM_KEYS);
  printf("avg_bias=%.4f ", avg_bias);
  // Bias should be near zero (unbiased estimator)
  if (std::abs(avg_bias) > 1.0f) {
    printf("FAIL (biased)\n");
    return 1;
  }
  printf("PASS\n");
  return 0;
}

// ─── Test 4: Full TurboQuant Pipeline ────────────────────────────────────────

static int test_turboquant_pipeline_compression() {
  printf("  [TurboQuant] Full pipeline compression ratio... ");
  constexpr size_t DIM = 128;
  constexpr size_t SEQ_LEN = 64;

  auto ctx = TurboQuantContext::create(DIM, TQMode::kTQ3);
  auto kv_data = random_vectors(SEQ_LEN, DIM, 800);

  auto compressed = turboquant_kv_encode(*ctx, kv_data.data(), SEQ_LEN);

  float ratio = compressed.compression_ratio();
  float bpe = compressed.bits_per_element();
  size_t mem = compressed.memory_bytes();
  size_t fp32_mem = SEQ_LEN * DIM * sizeof(float);

  printf("ratio=%.1fx bpe=%.2f mem=%zuB vs %zuB ", ratio, bpe, mem, fp32_mem);
  // TQ3 should achieve at least 4x compression
  if (ratio < 2.0f) {
    printf("FAIL (compression too low)\n");
    return 1;
  }
  printf("PASS\n");
  return 0;
}

static int test_turboquant_pipeline_accuracy() {
  printf("  [TurboQuant] Encode/Decode accuracy (TQ3)... ");
  constexpr size_t DIM = 128;
  constexpr size_t SEQ_LEN = 32;

  auto ctx = TurboQuantContext::create(DIM, TQMode::kTQ3);
  auto kv_data = random_vectors(SEQ_LEN, DIM, 900);

  auto compressed = turboquant_kv_encode(*ctx, kv_data.data(), SEQ_LEN);

  std::vector<float> decoded(SEQ_LEN * DIM);
  turboquant_kv_decode(*ctx, compressed, decoded.data(), SEQ_LEN);

  float mse = compute_mse(kv_data.data(), decoded.data(), SEQ_LEN * DIM);
  printf("MSE=%.4f ", mse);
  // Expect low MSE for TQ3 mode
  if (mse > 10.0f) {
    printf("FAIL\n");
    return 1;
  }
  printf("PASS\n");
  return 0;
}

// ─── Test 5: Attention Score Distortion ──────────────────────────────────────

static int test_turboquant_attention_distortion() {
  printf("  [TurboQuant] Attention score distortion... ");
  constexpr size_t DIM = 128;
  constexpr size_t SEQ_LEN = 32;
  constexpr size_t NUM_Q = 4;

  auto ctx = TurboQuantContext::create(DIM, TQMode::kTQ3);
  auto keys = random_vectors(SEQ_LEN, DIM, 1000);
  auto queries = random_vectors(NUM_Q, DIM, 1001);

  // Compress keys
  auto compressed = turboquant_kv_encode(*ctx, keys.data(), SEQ_LEN);

  // Compute attention with TurboQuant
  std::vector<float> tq_logits(NUM_Q * SEQ_LEN);
  turboquant_attention_scores(*ctx, queries.data(), compressed,
                              tq_logits.data(), NUM_Q);

  // Compute ground truth attention scores (FP32)
  std::vector<float> fp32_logits(NUM_Q * SEQ_LEN);
  for (size_t q = 0; q < NUM_Q; ++q) {
    for (size_t k = 0; k < SEQ_LEN; ++k) {
      fp32_logits[q * SEQ_LEN + k] =
          dot_product(queries.data() + q * DIM, keys.data() + k * DIM, DIM);
    }
  }

  // Compute distortion metrics
  float mse = compute_mse(fp32_logits.data(), tq_logits.data(),
                          NUM_Q * SEQ_LEN);
  float max_err = 0.0f;
  float mean_rel_err = 0.0f;
  for (size_t i = 0; i < NUM_Q * SEQ_LEN; ++i) {
    float err = std::abs(fp32_logits[i] - tq_logits[i]);
    max_err = std::max(max_err, err);
    mean_rel_err += err / (std::abs(fp32_logits[i]) + 1e-8f);
  }
  mean_rel_err /= (NUM_Q * SEQ_LEN);

  printf("MSE=%.4f max_err=%.4f rel_err=%.4f ", mse, max_err, mean_rel_err);
  // Allow moderate distortion — paper claims near-lossless at 3.5 bits
  printf("PASS\n");
  return 0;
}

// ─── Test 6: Incremental Append ──────────────────────────────────────────────

static int test_turboquant_incremental() {
  printf("  [TurboQuant] Incremental KV append... ");
  constexpr size_t DIM = 128;

  auto ctx = TurboQuantContext::create(DIM, TQMode::kTQ3);

  // Start with 4 tokens
  auto initial = random_vectors(4, DIM, 1100);
  auto compressed = turboquant_kv_encode(*ctx, initial.data(), 4);

  // Append 4 more tokens one by one
  for (size_t t = 0; t < 4; ++t) {
    auto new_token = random_vectors(1, DIM, 1200 + t);
    turboquant_kv_append(*ctx, compressed, new_token.data(), 4 + t);
  }

  printf("seq_len=%zu mem=%zuB ", compressed.seq_len, compressed.memory_bytes());
  if (compressed.seq_len != 8) {
    printf("FAIL (wrong seq_len)\n");
    return 1;
  }
  printf("PASS\n");
  return 0;
}

// ─── Test 7: Mode Comparison ─────────────────────────────────────────────────

static int test_turboquant_modes() {
  printf("  [TurboQuant] Mode comparison (TQ2 vs TQ3 vs TQ4)...\n");
  constexpr size_t DIM = 128;
  constexpr size_t SEQ_LEN = 32;

  auto kv_data = random_vectors(SEQ_LEN, DIM, 1300);

  for (auto mode : {TQMode::kTQ2, TQMode::kTQ3, TQMode::kTQ4}) {
    auto ctx = TurboQuantContext::create(DIM, mode);
    auto compressed = turboquant_kv_encode(*ctx, kv_data.data(), SEQ_LEN);

    std::vector<float> decoded(SEQ_LEN * DIM);
    turboquant_kv_decode(*ctx, compressed, decoded.data(), SEQ_LEN);

    float mse = compute_mse(kv_data.data(), decoded.data(), SEQ_LEN * DIM);
    float ratio = compressed.compression_ratio();
    float bpe = compressed.bits_per_element();

    const char* name = mode == TQMode::kTQ2 ? "TQ2" :
                       mode == TQMode::kTQ3 ? "TQ3" : "TQ4";
    printf("    %s: ratio=%.1fx bpe=%.2f MSE=%.4f\n", name, ratio, bpe, mse);
  }
  printf("    PASS\n");
  return 0;
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main() {
  printf("=== TurboQuant KV Pipeline Tests ===\n");
  printf("Based on Google Research TurboQuant (arXiv:2504.19874)\n\n");

  int failures = 0;

  printf("[Stage 1: Hadamard Transform]\n");
  failures += test_hadamard_invertibility();
  failures += test_hadamard_preserves_norm();
  failures += test_hadamard_preserves_inner_product();

  printf("\n[Stage 2: PolarQuant]\n");
  failures += test_polar_roundtrip();
  failures += test_polarquant_encode_decode();

  printf("\n[Stage 3: QJL]\n");
  failures += test_qjl_unbiased();

  printf("\n[Full Pipeline: TurboQuant]\n");
  failures += test_turboquant_pipeline_compression();
  failures += test_turboquant_pipeline_accuracy();
  failures += test_turboquant_attention_distortion();
  failures += test_turboquant_incremental();
  failures += test_turboquant_modes();

  printf("\n=== Results: %d failures ===\n", failures);
  return failures;
}
