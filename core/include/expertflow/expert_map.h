// ExpertFlow — MoE-Aware Inference Engine
// expert_map.h: GGUF tensor decomposition into shared vs expert weights
//
// Parses a GGUF model file and builds a map of:
//   - Shared tensors (attention, norms, embeddings, router, shared experts)
//   - Expert tensors indexed by (layer_id, expert_id, projection_type)
//
// This enables selective loading: shared weights stay on GPU permanently,
// expert weights are cached/streamed on demand.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace expertflow {

// Expert projection type within the MoE FFN block
enum class ExpertProj : uint8_t {
    kGateProj = 0,  // W1: gate projection (embed_dim → ffn_dim)
    kUpProj   = 1,  // W3: up projection   (embed_dim → ffn_dim)
    kDownProj = 2,  // W2: down projection  (ffn_dim → embed_dim)
    kCount    = 3
};

// Location of a tensor's raw data within the GGUF file
struct TensorLocation {
    std::string name;       // GGUF tensor name (e.g. "blk.5.ffn_gate_exps")
    uint64_t    offset;     // Byte offset from start of data section
    uint64_t    size_bytes; // Total tensor size in bytes
    uint32_t    n_dims;     // Number of dimensions
    uint64_t    ne[4];      // Shape (ggml order: ne[0]=cols, ne[1]=rows, ...)
    uint32_t    quant_type; // ggml_type enum value (e.g. IQ2_XXS = 10)
};

// Key for looking up a single expert's weights
struct ExpertKey {
    uint32_t layer_id;
    uint32_t expert_id;
    ExpertProj proj;

    bool operator==(const ExpertKey& o) const {
        return layer_id == o.layer_id &&
               expert_id == o.expert_id &&
               proj == o.proj;
    }
};

struct ExpertKeyHash {
    size_t operator()(const ExpertKey& k) const {
        // layer_id in high bits, expert_id in mid, proj in low
        return std::hash<uint64_t>{}(
            (static_cast<uint64_t>(k.layer_id) << 32) |
            (static_cast<uint64_t>(k.expert_id) << 2) |
            static_cast<uint64_t>(k.proj));
    }
};

// A single expert's weight slice within a merged 3D tensor
struct ExpertSlice {
    uint32_t    layer_id;
    uint32_t    expert_id;
    ExpertProj  proj;
    uint64_t    offset;       // Byte offset of this expert within the GGUF data section
    uint64_t    size_bytes;   // Size of this expert's weights in bytes
    uint64_t    rows;         // Matrix rows (output dim)
    uint64_t    cols;         // Matrix cols (input dim)
    uint32_t    quant_type;   // ggml_type
};

// Model architecture summary extracted from GGUF metadata
struct MoeArchitecture {
    std::string model_name;
    std::string architecture;   // e.g. "qwen35moe"
    uint32_t n_layers;          // Total transformer layers
    uint32_t n_experts;         // Experts per MoE layer
    uint32_t n_experts_used;    // Top-K experts activated per token
    uint32_t embed_dim;         // Model hidden dimension
    uint32_t expert_ffn_dim;    // Expert intermediate dimension
    uint32_t n_heads;           // Attention heads
    uint32_t n_kv_heads;        // KV attention heads (GQA)
    uint64_t context_length;    // Max sequence length
    uint32_t shared_expert_ffn_dim; // Shared expert FFN dim (0 if none)

    // Computed fields
    uint64_t expert_weight_bytes;    // Bytes per single expert (all 3 projections)
    uint64_t total_expert_bytes;     // Total expert weight pool size
    uint64_t shared_weight_bytes;    // Total shared (non-expert) weight size
    uint64_t active_bytes_per_layer; // Bytes read per layer (shared/layer + top-K experts)
    uint64_t active_bytes_per_token; // Bytes read per token (sum across all layers)

    // Speed estimates (tok/s) for given bandwidth
    double estimate_speed(double bw_bytes_per_sec) const {
        if (active_bytes_per_token == 0) return 0.0;
        return bw_bytes_per_sec / static_cast<double>(active_bytes_per_token);
    }
};

// Parsed GGUF model decomposed into shared + expert tensors
class ExpertMap {
public:
    ExpertMap() = default;
    ~ExpertMap() = default;

    // Non-copyable, movable
    ExpertMap(const ExpertMap&) = delete;
    ExpertMap& operator=(const ExpertMap&) = delete;
    ExpertMap(ExpertMap&&) = default;
    ExpertMap& operator=(ExpertMap&&) = default;

    // Parse a GGUF file and build the expert map.
    // Returns false if the file is not a valid MoE model.
    bool load(const std::string& gguf_path);

    // Is this a MoE model with experts?
    bool is_moe() const { return arch_.n_experts > 0; }

    // Architecture summary
    const MoeArchitecture& architecture() const { return arch_; }

    // All shared (non-expert) tensors
    const std::vector<TensorLocation>& shared_tensors() const { return shared_; }

    // All expert slices (flat list)
    const std::vector<ExpertSlice>& expert_slices() const { return experts_; }

    // Look up a specific expert's weight slice
    // Returns nullptr if not found
    const ExpertSlice* get_expert(uint32_t layer_id, uint32_t expert_id,
                                   ExpertProj proj) const;

    // Get all expert slices for a given layer
    std::vector<const ExpertSlice*> get_layer_experts(uint32_t layer_id) const;

    // Get the active expert slices for a set of routed expert IDs at one layer
    std::vector<const ExpertSlice*> get_active_experts(
        uint32_t layer_id, const std::vector<uint32_t>& expert_ids) const;

    // Raw GGUF data section offset (for mmap)
    uint64_t data_section_offset() const { return data_offset_; }

    // GGUF file path
    const std::string& gguf_path() const { return gguf_path_; }

    // Print summary to stdout
    void print_summary() const;

private:
    // Parse GGUF header and metadata
    bool parse_metadata(const uint8_t* data, size_t file_size);

    // Classify tensors as shared vs expert
    bool classify_tensors(const uint8_t* data, size_t file_size);

    std::string gguf_path_;
    MoeArchitecture arch_{};
    uint64_t data_offset_ = 0;  // Offset to start of tensor data in file

    std::vector<TensorLocation> shared_;   // Non-expert tensors
    std::vector<ExpertSlice> experts_;      // All expert weight slices

    // Fast lookup: (layer, expert, proj) → index into experts_
    std::unordered_map<ExpertKey, size_t, ExpertKeyHash> expert_index_;
};

}  // namespace expertflow
