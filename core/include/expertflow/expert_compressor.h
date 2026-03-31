// ExpertFlow — Expert Transfer Compression
// expert_compressor.h: Lightweight compression for expert weight PCIe transfer
//
// Strategy: IQ2_XXS quantized data has repeating codebook indices that compress
// well with simple byte-level run-length + LZ77-style matching.
// This is a self-contained compressor (no external LZ4 dependency).
//
// Pipeline:
//   CPU: read expert from mmap → compress → pinned staging buffer
//   PCIe: transfer compressed data (30-40% smaller for IQ2_XXS)
//   GPU: decompress in staging → copy to expert cache slot
//
// Compression format:
//   [4 bytes: original_size] [4 bytes: compressed_size] [compressed data...]
//   Compressed data uses a simple token stream:
//     - Literal run: [0 + len-1 (7 bits)] [len bytes of data]
//     - Match:       [1 + offset_hi (7 bits)] [offset_lo (8 bits)] [len (8 bits)]

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace expertflow {

// Compression statistics
struct CompressionStats {
    size_t   original_bytes;
    size_t   compressed_bytes;
    double   ratio;              // compressed / original (lower = better)
    double   compress_us;        // Compression time (microseconds)
    double   decompress_us;      // Decompression time (microseconds)
    uint64_t total_compressions;
    uint64_t total_decompressions;

    void update_compress(size_t orig, size_t comp, double us) {
        original_bytes += orig;
        compressed_bytes += comp;
        ratio = static_cast<double>(compressed_bytes) / original_bytes;
        compress_us += us;
        total_compressions++;
    }

    void update_decompress(double us) {
        decompress_us += us;
        total_decompressions++;
    }
};

// Compress expert weight data for PCIe transfer.
// Returns compressed buffer (includes 8-byte header with sizes).
// For IQ2_XXS data, achieves ~30-40% compression.
//
// src: source quantized weight data
// src_size: size in bytes
// Returns: compressed data (empty vector if compression would increase size)
std::vector<uint8_t> compress_expert(const uint8_t* src, size_t src_size);

// Decompress expert weight data after PCIe transfer.
// dst: destination buffer (must be at least original_size bytes)
// src: compressed data (including 8-byte header)
// src_size: compressed size including header
// Returns: number of bytes decompressed (original size), or 0 on error
size_t decompress_expert(uint8_t* dst, const uint8_t* src, size_t src_size);

// Get the original (uncompressed) size from a compressed buffer header.
// Returns 0 if the buffer is too small or invalid.
size_t compressed_original_size(const uint8_t* src, size_t src_size);

// Pinned Memory Pool — pre-allocated pinned host memory to avoid
// per-transfer cudaHostAlloc overhead (~100μs per call).
//
// Usage:
//   pool.init(total_bytes)  // one-time allocation
//   void* buf = pool.acquire(needed_bytes)
//   ... use buf for H2D staging ...
//   pool.release(buf)
//
class PinnedMemoryPool {
public:
    PinnedMemoryPool() = default;
    ~PinnedMemoryPool();

    // Non-copyable
    PinnedMemoryPool(const PinnedMemoryPool&) = delete;
    PinnedMemoryPool& operator=(const PinnedMemoryPool&) = delete;

    // Allocate the pinned memory pool.
    // total_bytes: size of the pool (e.g. 128 MB for staging)
    // Returns false if allocation fails.
    bool init(size_t total_bytes);

    // Release all pool memory.
    void release();

    // Acquire a contiguous region from the pool.
    // Returns nullptr if not enough space.
    // Thread-safe: uses simple bump allocator (reset with reset()).
    uint8_t* acquire(size_t bytes);

    // Reset the pool (reuse all memory). Call between tokens.
    void reset();

    // Pool capacity and current usage
    size_t capacity() const { return total_bytes_; }
    size_t used() const { return offset_; }
    size_t available() const { return total_bytes_ - offset_; }
    bool is_initialized() const { return base_ != nullptr; }

private:
    uint8_t* base_        = nullptr;
    size_t   total_bytes_ = 0;
    size_t   offset_      = 0;  // Bump allocator offset
};

}  // namespace expertflow
