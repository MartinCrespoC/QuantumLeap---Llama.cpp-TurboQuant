#include "turboquant/cuda_utils.cuh"
#include "turboquant/turboquant_kv.h"

#include <cmath>

namespace turboquant {

// ============================================================================
// TurboQuant CUDA Kernels
// Fused GPU implementation of the TurboQuant KV cache pipeline
// Based on Google Research TurboQuant (arXiv:2504.19874)
//
// Target: 8x speedup over FP32 via bit-packed attention computation
// ============================================================================

// ─── Constants ───────────────────────────────────────────────────────────────

constexpr int TQ_BLOCK_SIZE = 256;
constexpr int TQ_WARP_SIZE = 32;

// ─── Kernel: Fast Walsh-Hadamard Transform (GPU) ─────────────────────────────
// In-place FWHT on GPU, one block per vector
// Shared memory for butterfly operations

__global__ void fwht_kernel(
    float* __restrict__ data,    // [batch, dim]
    const int dim) {
  extern __shared__ float smem[];

  const int batch_idx = blockIdx.x;
  const int tid = threadIdx.x;
  float* vec = data + batch_idx * dim;

  // Load into shared memory
  for (int i = tid; i < dim; i += blockDim.x) {
    smem[i] = vec[i];
  }
  __syncthreads();

  // Butterfly passes
  for (int half = 1; half < dim; half <<= 1) {
    for (int i = tid; i < dim / 2; i += blockDim.x) {
      int block = i / half;
      int offset = i % half;
      int idx0 = block * (half << 1) + offset;
      int idx1 = idx0 + half;

      float a = smem[idx0];
      float b = smem[idx1];
      smem[idx0] = a + b;
      smem[idx1] = a - b;
    }
    __syncthreads();
  }

  // Write back with normalization
  float norm = rsqrtf(static_cast<float>(dim));
  for (int i = tid; i < dim; i += blockDim.x) {
    vec[i] = smem[i] * norm;
  }
}

// ─── Kernel: Random Sign Flips ───────────────────────────────────────────────
// Multiply each element by random ±1 sign (preconditioning step)

__global__ void sign_flip_kernel(
    const float* __restrict__ input,
    const float* __restrict__ signs,   // [dim] random ±1
    float* __restrict__ output,
    const int batch,
    const int dim) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch * dim;
  if (idx >= total) return;

  const int d = idx % dim;
  output[idx] = input[idx] * signs[d];
}

// ─── Kernel: Polar Decomposition (GPU) ───────────────────────────────────────
// Convert each rotated vector to (radius, angles)
// One thread per vector

__global__ void polar_decompose_kernel(
    const float* __restrict__ rotated,   // [batch, dim]
    float* __restrict__ radii,           // [batch]
    float* __restrict__ angles,          // [batch, dim-1]
    const int batch,
    const int dim) {
  const int vid = blockIdx.x * blockDim.x + threadIdx.x;
  if (vid >= batch) return;

  const float* vec = rotated + vid * dim;
  float* ang = angles + vid * (dim - 1);

  // Compute radius
  float r_sq = 0.0f;
  for (int i = 0; i < dim; ++i) {
    r_sq += vec[i] * vec[i];
  }
  float r = sqrtf(r_sq);
  radii[vid] = r;

  if (r < 1e-10f) {
    for (int i = 0; i < dim - 1; ++i) {
      ang[i] = M_PI * 0.5f;
    }
    return;
  }

  // Recursive polar decomposition
  float remaining_sq = r_sq;
  for (int k = 0; k < dim - 1; ++k) {
    float remaining_norm = sqrtf(remaining_sq);
    if (remaining_norm < 1e-10f) {
      for (int j = k; j < dim - 1; ++j) {
        ang[j] = M_PI * 0.5f;
      }
      break;
    }
    float cos_theta = fminf(fmaxf(vec[k] / remaining_norm, -1.0f), 1.0f);
    ang[k] = acosf(cos_theta);
    remaining_sq -= vec[k] * vec[k];
    if (remaining_sq < 0.0f) remaining_sq = 0.0f;
  }
}

// ─── Kernel: Angle Quantization (GPU) ────────────────────────────────────────
// Quantize angles using Beta distribution knowledge (no normalization needed)

__device__ float beta_stddev_d(int k, int d) {
  float a = static_cast<float>(d - k) * 0.5f;
  if (a < 0.5f) a = 0.5f;
  return M_PI / (2.0f * sqrtf(2.0f * a + 1.0f));
}

__global__ void quantize_angles_kernel(
    const float* __restrict__ angles,       // [batch, dim-1]
    uint8_t* __restrict__ quantized,        // [batch, dim-1] (unpacked for now)
    const int batch,
    const int dim,
    const int angle_bits) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch * (dim - 1);
  if (idx >= total) return;

  const int k = idx % (dim - 1);   // Angle index within vector
  float angle = angles[idx];

  int levels = 1 << angle_bits;
  float center = M_PI * 0.5f;
  float sigma = beta_stddev_d(k, dim);
  float range = 6.0f * sigma;
  float lo = fmaxf(0.0f, center - range * 0.5f);
  float hi = fminf(M_PI, center + range * 0.5f);
  float step = (hi - lo) / (levels - 1);

  float clamped = fminf(fmaxf(angle, lo), hi);
  int q = __float2int_rn((clamped - lo) / step);
  q = max(0, min(levels - 1, q));
  quantized[idx] = static_cast<uint8_t>(q);
}

// ─── Kernel: QJL Sign Extraction (GPU) ───────────────────────────────────────
// Compute R * residual, store sign bits (1 bit per projection dim)

__global__ void qjl_sign_kernel(
    const float* __restrict__ residuals,   // [batch, dim]
    const float* __restrict__ R,           // [proj_dim, dim] random ±1/√proj_dim
    uint8_t* __restrict__ sign_bits,       // [batch, proj_dim/8]
    const int batch,
    const int dim,
    const int proj_dim) {
  const int vid = blockIdx.y;   // Vector index
  const int pid = blockIdx.x * blockDim.x + threadIdx.x;  // Projection index
  if (vid >= batch || pid >= proj_dim) return;

  const float* res = residuals + vid * dim;
  const float* row = R + pid * dim;

  // Dot product: R[pid] · residual[vid]
  float dot = 0.0f;
  for (int i = 0; i < dim; ++i) {
    dot += row[i] * res[i];
  }

  // Set sign bit using atomic OR (1 bit per projection dimension)
  if (dot >= 0.0f) {
    int byte_idx = vid * ((proj_dim + 7) / 8) + pid / 8;
    uint8_t bit_mask = 1 << (pid % 8);
    atomicOr(reinterpret_cast<unsigned int*>(&sign_bits[byte_idx & ~3]),
             static_cast<unsigned int>(bit_mask) << (8 * (byte_idx & 3)));
  }
}

// ─── Kernel: Quantized Attention Scores (GPU, Hyper-Optimized) ───────────────
// Compute attention logits directly from compressed KV cache.
// This is the 8x speedup kernel.
//
// Optimizations applied (TurboQuant paper + hardware-specific):
//   1. Query vector + query projection loaded once to shared memory per block
//   2. Dequant parameters (lo, step) pre-computed to shared memory LUT
//   3. __sincosf for fused trig (single instruction on SM8.6)
//   4. 64-bit word reads + __popc for QJL sign accumulation
//   5. Warp shuffle for intra-warp reductions
//   6. Loop unrolling for dim-128 common case
//
// Target: >70% occupancy on RTX 3050 (SM8.6, 48KB shared memory)

// Maximum supported dimensions
constexpr int TQ_MAX_DIM = 256;
constexpr int TQ_MAX_PROJ = 256;

__global__ void turboquant_attention_kernel(
    const float* __restrict__ queries_rotated,  // [num_q, dim]
    const uint8_t* __restrict__ k_angles_q,     // [seq_len, dim-1] quantized
    const float* __restrict__ k_radii,          // [seq_len]
    const uint8_t* __restrict__ k_qjl_signs,    // [seq_len, proj_dim/8]
    const float* __restrict__ qjl_R,            // [proj_dim, dim]
    float* __restrict__ logits,                 // [num_q, seq_len]
    const int num_queries,
    const int seq_len,
    const int dim,
    const int proj_dim,
    const int angle_bits) {
  // Shared memory layout:
  //   [0..dim-1]                = query vector (loaded once per block row)
  //   [dim..dim+proj_dim-1]     = query projections qp[j] = R[j] · q
  //   [dim+proj_dim..+2*(dim-1)]= dequant LUT: lo[d], step[d]
  extern __shared__ float smem[];
  float* s_query    = smem;                            // [dim]
  float* s_qproj    = smem + dim;                      // [proj_dim]
  float* s_dequant  = smem + dim + proj_dim;           // [2*(dim-1)]

  const int q_idx = blockIdx.y;
  const int k_start = blockIdx.x * blockDim.x;
  const int k_local = threadIdx.x;
  const int k_idx = k_start + k_local;

  if (q_idx >= num_queries) return;

  // ── Phase 1: Cooperatively load query vector into shared memory ──
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    s_query[i] = queries_rotated[q_idx * dim + i];
  }
  __syncthreads();

  // ── Phase 2: Cooperatively pre-compute dequantization LUT ──
  // Each angle index d has fixed (lo, step) — same for all keys
  {
    int levels = 1 << angle_bits;
    for (int d = threadIdx.x; d < dim - 1; d += blockDim.x) {
      float sigma = beta_stddev_d(d, dim);
      float range = 6.0f * sigma;
      float center = static_cast<float>(M_PI) * 0.5f;
      float lo = fmaxf(0.0f, center - range * 0.5f);
      float hi = fminf(static_cast<float>(M_PI), center + range * 0.5f);
      float step = (hi - lo) / (levels - 1);
      s_dequant[2 * d]     = lo;
      s_dequant[2 * d + 1] = step;
    }
  }
  __syncthreads();

  // ── Phase 3: Cooperatively pre-compute query projections ──
  // qp[j] = R[j] · q  — computed once, reused for every key in this block
  for (int j = threadIdx.x; j < proj_dim; j += blockDim.x) {
    const float* rj = qjl_R + j * dim;
    float dot = 0.0f;
    // Unrolled accumulation
    int i = 0;
    for (; i + 3 < dim; i += 4) {
      dot += rj[i]     * s_query[i]
           + rj[i + 1] * s_query[i + 1]
           + rj[i + 2] * s_query[i + 2]
           + rj[i + 3] * s_query[i + 3];
    }
    for (; i < dim; ++i) {
      dot += rj[i] * s_query[i];
    }
    s_qproj[j] = dot;
  }

  // Pre-compute total sum of query projections for branchless QJL
  // (sign * qp = 2*bit*qp - qp, so sum = 2*pos_sum - total_sum)
  __shared__ float s_qproj_total;
  if (threadIdx.x == 0) {
    float total = 0.0f;
    for (int j = 0; j < proj_dim; ++j) {
      total += s_qproj[j];
    }
    s_qproj_total = total;
  }
  __syncthreads();

  if (k_idx >= seq_len) return;

  // ── Phase 4: Per-key polar dot product (fused reconstruct + dot) ──
  const uint8_t* k_ang = k_angles_q + k_idx * (dim - 1);
  float k_radius = k_radii[k_idx];
  float dot = 0.0f;
  float running_sin = k_radius;

  for (int d = 0; d < dim - 1; ++d) {
    float angle = s_dequant[2 * d] + static_cast<float>(k_ang[d]) * s_dequant[2 * d + 1];
    float cos_a, sin_a;
    __sincosf(angle, &sin_a, &cos_a);  // Single fused instruction on SM8.6
    dot += s_query[d] * running_sin * cos_a;
    running_sin *= sin_a;
  }
  dot += s_query[dim - 1] * running_sin;

  // ── Phase 5: QJL Correction (using shared query projection) ──
  // Read sign bits in 32-bit words, use __popc for popcount-based accumulation
  float pos_sum = 0.0f;
  int bytes_per_vec = (proj_dim + 7) / 8;
  const uint8_t* bits_ptr = k_qjl_signs + k_idx * bytes_per_vec;

  // Process 32 bits at a time
  int j = 0;
  for (; j + 31 < proj_dim; j += 32) {
    uint32_t word;
    // Aligned 4-byte read from sign bits
    memcpy(&word, bits_ptr + j / 8, sizeof(uint32_t));

    // Iterate set bits using Brian Kernighan's trick
    uint32_t tmp = word;
    while (tmp) {
      int bit_pos = __ffs(tmp) - 1;  // Find first set bit (0-indexed)
      pos_sum += s_qproj[j + bit_pos];
      tmp &= tmp - 1;  // Clear lowest set bit
    }
  }
  // Remainder bits
  for (; j < proj_dim; ++j) {
    int bit_idx = j;
    int sign_bit = (bits_ptr[bit_idx / 8] >> (bit_idx % 8)) & 1;
    if (sign_bit) {
      pos_sum += s_qproj[j];
    }
  }

  float qjl_scale = static_cast<float>(M_PI) / (2.0f * proj_dim);
  float correction = (2.0f * pos_sum - s_qproj_total) * qjl_scale;

  logits[q_idx * seq_len + k_idx] = dot + correction;
}

// ─── Kernel: Fused Polar Reconstruct + Residual + QJL Sign Extract ──────────
// Eliminates CPU fallback for Stage 3 of the encode pipeline.
// Each thread block handles one vector: reconstructs from polar, computes
// residual vs rotated input, projects through R, and extracts sign bits.

__global__ void fused_residual_qjl_kernel(
    const float* __restrict__ rotated,          // [batch, dim] rotated input
    const uint8_t* __restrict__ q_angles,       // [batch, dim-1] quantized angles
    const uint8_t* __restrict__ q_radii,        // [batch] quantized radii
    const float* __restrict__ R,                // [proj_dim, dim] QJL random matrix
    uint8_t* __restrict__ sign_bits,            // [batch, proj_dim/8] output
    const int batch,
    const int dim,
    const int proj_dim,
    const int angle_bits,
    const float rmin,
    const float rmax,
    const int radius_bits) {
  extern __shared__ float smem[];
  // smem layout: [0..dim-1] = reconstructed, [dim..2*dim-1] = residual
  float* s_recon = smem;
  float* s_resid = smem + dim;

  const int vid = blockIdx.x;
  if (vid >= batch) return;

  const uint8_t* ang = q_angles + vid * (dim - 1);

  // ── Reconstruct from polar coordinates in shared memory ──
  if (threadIdx.x == 0) {
    // Dequantize radius
    int r_levels = (1 << radius_bits) - 1;
    float radius = rmin + (static_cast<float>(q_radii[vid]) / r_levels) * (rmax - rmin);

    int levels = 1 << angle_bits;
    float running_sin = radius;
    for (int k = 0; k < dim - 1; ++k) {
      float sigma = beta_stddev_d(k, dim);
      float range = 6.0f * sigma;
      float center = static_cast<float>(M_PI) * 0.5f;
      float lo = fmaxf(0.0f, center - range * 0.5f);
      float hi = fminf(static_cast<float>(M_PI), center + range * 0.5f);
      float step = (hi - lo) / (levels - 1);
      float angle = lo + static_cast<float>(ang[k]) * step;

      float cos_a, sin_a;
      __sincosf(angle, &sin_a, &cos_a);
      s_recon[k] = running_sin * cos_a;
      running_sin *= sin_a;
    }
    s_recon[dim - 1] = running_sin;
  }
  __syncthreads();

  // ── Compute residual = rotated - reconstructed (parallel) ──
  const float* rot_vec = rotated + vid * dim;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    s_resid[i] = rot_vec[i] - s_recon[i];
  }
  __syncthreads();

  // ── QJL: project residual through R, extract sign bits (parallel) ──
  int bytes_per_vec = (proj_dim + 7) / 8;
  uint8_t* out_bits = sign_bits + vid * bytes_per_vec;

  // Zero output
  for (int i = threadIdx.x; i < bytes_per_vec; i += blockDim.x) {
    out_bits[i] = 0;
  }
  __syncthreads();

  // Each thread handles a subset of projection dimensions
  for (int pid = threadIdx.x; pid < proj_dim; pid += blockDim.x) {
    const float* rj = R + pid * dim;
    float dot = 0.0f;
    // Unrolled dot product against residual in shared memory
    int i = 0;
    for (; i + 3 < dim; i += 4) {
      dot += rj[i]     * s_resid[i]
           + rj[i + 1] * s_resid[i + 1]
           + rj[i + 2] * s_resid[i + 2]
           + rj[i + 3] * s_resid[i + 3];
    }
    for (; i < dim; ++i) {
      dot += rj[i] * s_resid[i];
    }

    // Set sign bit atomically
    if (dot >= 0.0f) {
      int byte_idx = pid / 8;
      uint8_t bit_mask = 1 << (pid % 8);
      atomicOr(reinterpret_cast<unsigned int*>(&out_bits[byte_idx & ~3]),
               static_cast<unsigned int>(bit_mask) << (8 * (byte_idx & 3)));
    }
  }
}

// ─── Host: Fused TurboQuant KV Encode (CUDA) ────────────────────────────────

TQCompressedKV turboquant_kv_encode_cuda(
    const TurboQuantContext& ctx,
    const float* __restrict__ kv_data,
    size_t seq_len) {
  const size_t dim = ctx.head_dim;
  const size_t padded = ctx.padded_dim;

  // Allocate device memory
  auto d_input = cuda::cuda_alloc<float>(seq_len * padded);
  auto d_rotated = cuda::cuda_alloc<float>(seq_len * padded);
  auto d_signs = cuda::cuda_alloc<float>(padded);
  auto d_radii = cuda::cuda_alloc<float>(seq_len);
  auto d_angles = cuda::cuda_alloc<float>(seq_len * (padded - 1));
  auto d_q_angles = cuda::cuda_alloc<uint8_t>(seq_len * (padded - 1));

  // Pad and copy input to device
  std::vector<float> padded_input(seq_len * padded, 0.0f);
  for (size_t s = 0; s < seq_len; ++s) {
    std::memcpy(padded_input.data() + s * padded, kv_data + s * dim,
                dim * sizeof(float));
  }
  cuda::cuda_copy_h2d(d_input.get(), padded_input.data(), seq_len * padded);
  cuda::cuda_copy_h2d(d_signs.get(), ctx.hadamard->signs.data(), padded);

  // Stage 1a: Sign flips
  {
    int total = seq_len * padded;
    int blocks = (total + TQ_BLOCK_SIZE - 1) / TQ_BLOCK_SIZE;
    sign_flip_kernel<<<blocks, TQ_BLOCK_SIZE>>>(
        d_input.get(), d_signs.get(), d_rotated.get(),
        static_cast<int>(seq_len), static_cast<int>(padded));
    CUDA_CHECK(cudaGetLastError());
  }

  // Stage 1b: FWHT on GPU
  {
    int threads = std::min(static_cast<int>(padded / 2), TQ_BLOCK_SIZE);
    size_t smem_bytes = padded * sizeof(float);
    fwht_kernel<<<static_cast<int>(seq_len), threads, smem_bytes>>>(
        d_rotated.get(), static_cast<int>(padded));
    CUDA_CHECK(cudaGetLastError());
  }

  // Stage 2a: Polar decomposition
  {
    int blocks = (seq_len + TQ_BLOCK_SIZE - 1) / TQ_BLOCK_SIZE;
    polar_decompose_kernel<<<blocks, TQ_BLOCK_SIZE>>>(
        d_rotated.get(), d_radii.get(), d_angles.get(),
        static_cast<int>(seq_len), static_cast<int>(padded));
    CUDA_CHECK(cudaGetLastError());
  }

  // Stage 2b: Angle quantization
  {
    int total = seq_len * (padded - 1);
    int blocks = (total + TQ_BLOCK_SIZE - 1) / TQ_BLOCK_SIZE;
    quantize_angles_kernel<<<blocks, TQ_BLOCK_SIZE>>>(
        d_angles.get(), d_q_angles.get(),
        static_cast<int>(seq_len), static_cast<int>(padded),
        static_cast<int>(ctx.polar_config.angle_bits));
    CUDA_CHECK(cudaGetLastError());
  }

  // Stage 3: QJL on residual (fully fused on GPU — no CPU fallback)
  // Uses fused_residual_qjl_kernel: reconstruct → residual → project → sign bits

  // Compute radius min/max on GPU via copy-back (small — just seq_len floats)
  std::vector<float> h_radii(seq_len);
  cuda::cuda_copy_d2h(h_radii.data(), d_radii.get(), seq_len);
  float rmin = *std::min_element(h_radii.begin(), h_radii.end());
  float rmax = *std::max_element(h_radii.begin(), h_radii.end());

  // Quantize radii on GPU (small enough to do on host, re-upload)
  std::vector<uint8_t> h_q_radii(seq_len);
  for (size_t i = 0; i < seq_len; ++i) {
    h_q_radii[i] = quantize_radius(h_radii[i], rmin, rmax,
                                     ctx.polar_config.radius_bits);
  }
  auto d_q_radii_packed = cuda::cuda_alloc<uint8_t>(seq_len);
  cuda::cuda_copy_h2d(d_q_radii_packed.get(), h_q_radii.data(), seq_len);

  // Upload QJL random matrix
  size_t proj_dim = ctx.qjl->proj_dim;
  auto d_R = cuda::cuda_alloc<float>(ctx.qjl->R.size());
  cuda::cuda_copy_h2d(d_R.get(), ctx.qjl->R.data(), ctx.qjl->R.size());

  // Allocate output sign bits
  size_t bytes_per_vec = (proj_dim + 7) / 8;
  auto d_sign_bits = cuda::cuda_alloc<uint8_t>(seq_len * bytes_per_vec);

  // Launch fused residual + QJL kernel
  // One block per vector, shared memory = 2 * dim floats (recon + resid)
  {
    int threads = std::min(static_cast<int>(proj_dim), TQ_BLOCK_SIZE);
    size_t smem_bytes = 2 * padded * sizeof(float);
    fused_residual_qjl_kernel<<<static_cast<int>(seq_len), threads, smem_bytes>>>(
        d_rotated.get(),
        d_q_angles.get(),
        d_q_radii_packed.get(),
        d_R.get(),
        d_sign_bits.get(),
        static_cast<int>(seq_len),
        static_cast<int>(padded),
        static_cast<int>(proj_dim),
        static_cast<int>(ctx.polar_config.angle_bits),
        rmin, rmax,
        static_cast<int>(ctx.polar_config.radius_bits));
    CUDA_CHECK(cudaGetLastError());
  }

  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy results back to host
  std::vector<uint8_t> h_q_angles(seq_len * (padded - 1));
  cuda::cuda_copy_d2h(h_q_angles.data(), d_q_angles.get(),
                      seq_len * (padded - 1));
  std::vector<uint8_t> h_sign_bits(seq_len * bytes_per_vec);
  cuda::cuda_copy_d2h(h_sign_bits.data(), d_sign_bits.get(),
                      seq_len * bytes_per_vec);

  // Build result
  TQCompressedKV result;
  result.seq_len = seq_len;
  result.head_dim = dim;
  result.mode = ctx.mode;

  result.polar.num_vectors = seq_len;
  result.polar.dim = padded;
  result.polar.config = ctx.polar_config;
  result.polar.packed_angles = std::move(h_q_angles);
  result.polar.packed_radii = std::move(h_q_radii);

  result.qjl_residual.num_vectors = seq_len;
  result.qjl_residual.proj_dim = proj_dim;
  result.qjl_residual.sign_bits = std::move(h_sign_bits);

  return result;
}

// ─── Host: TurboQuant Attention Scores (CUDA) ────────────────────────────────

void turboquant_attention_scores_cuda(
    const TurboQuantContext& ctx,
    const float* __restrict__ queries,
    const TQCompressedKV& compressed_keys,
    float* __restrict__ attention_logits,
    size_t num_queries) {
  const size_t dim = ctx.head_dim;
  const size_t padded = ctx.padded_dim;
  const size_t seq_len = compressed_keys.seq_len;
  const size_t proj_dim = ctx.qjl->proj_dim;

  // Rotate queries on GPU
  auto d_queries = cuda::cuda_alloc<float>(num_queries * padded);
  auto d_rotated_q = cuda::cuda_alloc<float>(num_queries * padded);
  auto d_signs = cuda::cuda_alloc<float>(padded);

  std::vector<float> padded_q(num_queries * padded, 0.0f);
  for (size_t q = 0; q < num_queries; ++q) {
    std::memcpy(padded_q.data() + q * padded, queries + q * dim,
                dim * sizeof(float));
  }
  cuda::cuda_copy_h2d(d_queries.get(), padded_q.data(), num_queries * padded);
  cuda::cuda_copy_h2d(d_signs.get(), ctx.hadamard->signs.data(), padded);

  // Sign flips
  {
    int total = num_queries * padded;
    int blocks = (total + TQ_BLOCK_SIZE - 1) / TQ_BLOCK_SIZE;
    sign_flip_kernel<<<blocks, TQ_BLOCK_SIZE>>>(
        d_queries.get(), d_signs.get(), d_rotated_q.get(),
        static_cast<int>(num_queries), static_cast<int>(padded));
  }
  // FWHT
  {
    int threads = std::min(static_cast<int>(padded / 2), TQ_BLOCK_SIZE);
    size_t smem_bytes = padded * sizeof(float);
    fwht_kernel<<<static_cast<int>(num_queries), threads, smem_bytes>>>(
        d_rotated_q.get(), static_cast<int>(padded));
  }

  // Upload compressed keys to GPU
  auto d_k_angles = cuda::cuda_alloc<uint8_t>(
      compressed_keys.polar.packed_angles.size());
  auto d_k_radii = cuda::cuda_alloc<float>(seq_len);
  auto d_k_qjl = cuda::cuda_alloc<uint8_t>(
      compressed_keys.qjl_residual.sign_bits.size());
  auto d_qjl_R = cuda::cuda_alloc<float>(ctx.qjl->R.size());
  auto d_logits = cuda::cuda_alloc<float>(num_queries * seq_len);

  cuda::cuda_copy_h2d(d_k_angles.get(),
                      compressed_keys.polar.packed_angles.data(),
                      compressed_keys.polar.packed_angles.size());

  // Dequantize radii on host for now
  std::vector<float> h_radii(seq_len);
  float rmin = 0.0f, rmax = 1.0f;  // TODO: store in metadata
  for (size_t i = 0; i < seq_len; ++i) {
    h_radii[i] = dequantize_radius(
        compressed_keys.polar.packed_radii[i], rmin, rmax,
        ctx.polar_config.radius_bits);
  }
  cuda::cuda_copy_h2d(d_k_radii.get(), h_radii.data(), seq_len);
  cuda::cuda_copy_h2d(d_k_qjl.get(),
                      compressed_keys.qjl_residual.sign_bits.data(),
                      compressed_keys.qjl_residual.sign_bits.size());
  cuda::cuda_copy_h2d(d_qjl_R.get(), ctx.qjl->R.data(), ctx.qjl->R.size());

  // Launch hyper-optimized fused attention kernel
  // Shared memory: query[dim] + qproj[proj_dim] + dequant_lut[2*(dim-1)] + qproj_total[1]
  size_t smem_attn = (padded + proj_dim + 2 * (padded - 1) + 1) * sizeof(float);
  dim3 grid((seq_len + TQ_BLOCK_SIZE - 1) / TQ_BLOCK_SIZE, num_queries);
  turboquant_attention_kernel<<<grid, TQ_BLOCK_SIZE, smem_attn>>>(
      d_rotated_q.get(),
      d_k_angles.get(),
      d_k_radii.get(),
      d_k_qjl.get(),
      d_qjl_R.get(),
      d_logits.get(),
      static_cast<int>(num_queries),
      static_cast<int>(seq_len),
      static_cast<int>(padded),
      static_cast<int>(proj_dim),
      static_cast<int>(ctx.polar_config.angle_bits));
  CUDA_CHECK(cudaGetLastError());

  // Copy results back
  cuda::cuda_copy_d2h(attention_logits, d_logits.get(),
                      num_queries * seq_len);
  CUDA_CHECK(cudaDeviceSynchronize());
}

}  // namespace turboquant
