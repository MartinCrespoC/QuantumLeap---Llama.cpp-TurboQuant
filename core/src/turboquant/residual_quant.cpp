#include "turboquant/turboquant.h"
#include "turboquant/simd_utils.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace turboquant {

namespace {

// Compute scale and zero point for a group of values
void compute_group_params(const float* data, size_t n,
                          QuantBits bits, float& scale, float& zero_point) {
  float min_val = *std::min_element(data, data + n);
  float max_val = *std::max_element(data, data + n);

  int max_quant = (1 << static_cast<int>(bits)) - 1;
  float range = max_val - min_val;

  if (range < 1e-8f) {
    scale = 1.0f;
    zero_point = 0.0f;
    return;
  }

  scale = range / static_cast<float>(max_quant);
  zero_point = min_val;
}

// Pack INT2 values: 4 values per byte
void pack_int2(const int* values, uint8_t* packed, size_t n) {
  for (size_t i = 0; i < n; i += 4) {
    uint8_t byte = 0;
    for (size_t j = 0; j < 4 && (i + j) < n; ++j) {
      byte |= (static_cast<uint8_t>(values[i + j] & 0x3)) << (j * 2);
    }
    packed[i / 4] = byte;
  }
}

// Pack INT4 values: 2 values per byte
void pack_int4(const int* values, uint8_t* packed, size_t n) {
  for (size_t i = 0; i < n; i += 2) {
    uint8_t lo = static_cast<uint8_t>(values[i] & 0xF);
    uint8_t hi = (i + 1 < n) ? static_cast<uint8_t>(values[i + 1] & 0xF) : 0;
    packed[i / 2] = lo | (hi << 4);
  }
}

}  // namespace

QuantResult residual_quantize(
    const float* data, size_t n, QuantBits bits,
    size_t group_size, int num_iterations) {
  QuantResult result;
  result.meta.num_elements = n;
  result.meta.bits = bits;
  result.meta.method = QuantMethod::kTurboQuant;
  result.meta.group_size = group_size;
  result.meta.num_groups = (n + group_size - 1) / group_size;

  int max_quant = (1 << static_cast<int>(bits)) - 1;

  // Allocate scales and zero points
  size_t num_groups = result.meta.num_groups;
  std::vector<float> scales(num_groups);
  std::vector<float> zero_points(num_groups);
  std::vector<int> quantized(n);

  // Working copy of residuals
  std::vector<float> residual(data, data + n);

  // Iterative residual quantization
  for (int iter = 0; iter < num_iterations; ++iter) {
    for (size_t g = 0; g < num_groups; ++g) {
      size_t start = g * group_size;
      size_t end = std::min(start + group_size, n);
      size_t len = end - start;

      float scale, zero;
      compute_group_params(residual.data() + start, len, bits, scale, zero);

      scales[g] = scale;
      zero_points[g] = zero;

      // Quantize this group
      for (size_t i = start; i < end; ++i) {
        float val = (residual[i] - zero) / (scale + 1e-10f);
        int q = static_cast<int>(std::round(val));
        q = std::clamp(q, 0, max_quant);
        quantized[i] = q;

        // Update residual for next iteration
        float dequantized = q * scale + zero;
        residual[i] = data[i] - dequantized;
      }
    }
  }

  // Pack quantized values
  size_t packed_size;
  if (bits == QuantBits::kInt2) {
    packed_size = (n + 3) / 4;
    result.data.resize(packed_size);
    pack_int2(quantized.data(), result.data.data(), n);
  } else if (bits == QuantBits::kInt4) {
    packed_size = (n + 1) / 2;
    result.data.resize(packed_size);
    pack_int4(quantized.data(), result.data.data(), n);
  } else {
    packed_size = n;
    result.data.resize(packed_size);
    for (size_t i = 0; i < n; ++i) {
      result.data[i] = static_cast<uint8_t>(quantized[i]);
    }
  }

  // Compute error metrics
  double mse_sum = 0.0;
  float max_err = 0.0f;
  for (size_t g = 0; g < num_groups; ++g) {
    size_t start = g * group_size;
    size_t end = std::min(start + group_size, n);
    for (size_t i = start; i < end; ++i) {
      float dequant = quantized[i] * scales[g] + zero_points[g];
      float err = std::abs(data[i] - dequant);
      mse_sum += static_cast<double>(err * err);
      max_err = std::max(max_err, err);
    }
  }
  result.mse = static_cast<float>(mse_sum / n);
  result.max_error = max_err;

  // Store scales (caller must manage lifetime)
  result.meta.scales = new float[num_groups];
  result.meta.zero_points = new float[num_groups];
  std::copy(scales.begin(), scales.end(), result.meta.scales);
  std::copy(zero_points.begin(), zero_points.end(), result.meta.zero_points);

  return result;
}

QuantResult turboquant_encode(
    const float* data, size_t n, QuantBits bits, size_t group_size) {
  // Stage 1: PolarQuant transform on pairs
  size_t pair_count = n / 2;
  std::vector<float> magnitudes(pair_count);
  std::vector<float> angles(pair_count);

  polar_transform(data, data + pair_count, magnitudes.data(), angles.data(),
                  pair_count);

  // Stage 2: Residual quantization on transformed data
  // Quantize magnitudes and angles separately for better compression
  auto mag_result = residual_quantize(magnitudes.data(), pair_count, bits,
                                       group_size, 3);
  auto ang_result = residual_quantize(angles.data(), pair_count, bits,
                                       group_size, 3);

  // Combine results
  QuantResult combined;
  combined.meta = mag_result.meta;
  combined.meta.method = QuantMethod::kTurboQuant;
  combined.meta.num_elements = n;
  combined.meta.magnitudes = new float[pair_count];
  combined.meta.angles = new float[pair_count];
  std::copy(magnitudes.begin(), magnitudes.end(), combined.meta.magnitudes);
  std::copy(angles.begin(), angles.end(), combined.meta.angles);

  // Interleave packed magnitude and angle data
  combined.data.reserve(mag_result.data.size() + ang_result.data.size());
  combined.data.insert(combined.data.end(), mag_result.data.begin(),
                       mag_result.data.end());
  combined.data.insert(combined.data.end(), ang_result.data.begin(),
                       ang_result.data.end());

  combined.mse = (mag_result.mse + ang_result.mse) / 2.0f;
  combined.max_error = std::max(mag_result.max_error, ang_result.max_error);

  return combined;
}

void turboquant_decode(const QuantResult& quantized, float* output, size_t n) {
  // TODO: Implement full TurboQuant decoding
  // 1. Separate magnitude and angle packed data
  // 2. Dequantize magnitudes and angles
  // 3. Inverse polar transform: x = mag*cos(angle), y = mag*sin(angle)
  (void)quantized;
  (void)output;
  (void)n;
}

}  // namespace turboquant
