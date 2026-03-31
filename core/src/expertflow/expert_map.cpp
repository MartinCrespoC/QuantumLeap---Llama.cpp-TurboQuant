// ExpertFlow — expert_map.cpp
// GGUF parser: decomposes model into shared tensors + per-expert weight slices

#include "expertflow/expert_map.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <regex>
#include <string>
#include <vector>

namespace expertflow {

// GGUF format constants
static constexpr uint32_t GGUF_MAGIC   = 0x46554747;  // "GGUF" in little-endian
static constexpr uint32_t GGUF_VERSION = 3;

// GGUF metadata value types
enum GGUFType : uint32_t {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

// Block byte sizes for GGML quantization types (from ggml-common.h struct sizes)
// type_size = sizeof(block_xxx) in bytes
// blck_size = number of elements per block
//
// Enum values from ggml.h: GGML_TYPE_F32=0, F16=1, Q4_0=2, Q4_1=3, ...
// Struct sizes computed from ggml-common.h static_asserts

struct GGMLTypeInfo {
    uint64_t type_size;  // sizeof(block) in bytes
    uint64_t blck_size;  // elements per block
};

// QK_K=256, ggml_half=2, K_SCALE_SIZE=12, QK4_NL=32, QK4_0=32, etc.
static GGMLTypeInfo ggml_type_info(uint32_t type) {
    switch (type) {
        // Unquantized
        case 0:  return {4, 1};       // F32:  4 bytes/elem
        case 1:  return {2, 1};       // F16:  2 bytes/elem

        // Basic quants (block of 32)
        case 2:  return {18, 32};     // Q4_0: ggml_half + 32/2 = 2+16 = 18
        case 3:  return {20, 32};     // Q4_1: 2*ggml_half + 32/2 = 4+16 = 20
        case 6:  return {22, 32};     // Q5_0: ggml_half + uint32 + 32/2 = 2+4+16 = 22
        case 7:  return {24, 32};     // Q5_1: 2*ggml_half + uint32 + 32/2 = 4+4+16 = 24
        case 8:  return {34, 32};     // Q8_0: ggml_half + 32 = 2+32 = 34
        case 9:  return {36, 32};     // Q8_1: 2*ggml_half + 32 = 4+32 = 36

        // K-quants (block of 256)
        case 10: return {84, 256};    // Q2_K: 2*ggml_half + 256/16 + 256/4 = 4+16+64 = 84
        case 11: return {110, 256};   // Q3_K: ggml_half + 256/4 + 256/8 + 12 = 2+64+32+12 = 110
        case 12: return {144, 256};   // Q4_K: 2*ggml_half + 12 + 256/2 = 4+12+128 = 144
        case 13: return {176, 256};   // Q5_K: 2*ggml_half + 12 + 256/2 + 256/8 = 4+12+128+32 = 176
        case 14: return {210, 256};   // Q6_K: ggml_half + 256/16 + 3*256/4 = 2+16+192 = 210
        case 15: return {292, 256};   // Q8_K: float + 256 + 256/16*2 = 4+256+32 = 292

        // IQ quants (block of 256 except IQ4_NL which is 32)
        case 16: return {66, 256};    // IQ2_XXS: ggml_half + 256/8*2 = 2+64 = 66
        case 17: return {74, 256};    // IQ2_XS:  ggml_half + 256/8*2 + 256/32 = 2+64+8 = 74
        case 18: return {98, 256};    // IQ3_XXS: ggml_half + 3*(256/8) = 2+96 = 98
        case 19: return {50, 256};    // IQ1_S:   ggml_half + 256/8 + 256/16 = 2+32+16 = 50
        case 20: return {18, 32};     // IQ4_NL:  ggml_half + 32/2 = 2+16 = 18
        case 21: return {110, 256};   // IQ3_S:   ggml_half + 13*(256/32) + 2 = 2+104+2 ≈ 110
        case 22: return {82, 256};    // IQ2_S:   ggml_half + 256/4 + 256/16 = 2+64+16 = 82
        case 23: return {136, 256};   // IQ4_XS:  ggml_half + uint16 + 256/64 + 256/2 = 2+2+4+128 = 136

        // Integer types
        case 24: return {1, 1};       // I8
        case 25: return {2, 1};       // I16
        case 26: return {4, 1};       // I32
        case 27: return {8, 1};       // I64
        case 28: return {8, 1};       // F64

        case 29: return {56, 256};    // IQ1_M:  256/8 + 256/16 + 256/32 = 32+16+8 = 56
        case 30: return {2, 1};       // BF16:   2 bytes/elem

        // TQ types (block of 256)
        case 34: return {54, 256};    // TQ1_0: ggml_half + 256/64 + (256-16)/5 = 2+4+48 = 54
        case 35: return {66, 256};    // TQ2_0: ggml_half + 256/4 = 2+64 = 66

        // MX/NV FP4
        case 39: return {17, 32};     // MXFP4: uint8 + 32/2 = 1+16 = 17
        case 40: return {36, 64};     // NVFP4: uint8*4 + 64/2 = 4+32 = 36

        default: return {2, 1};       // Fallback
    }
}

// Helper: read little-endian values from buffer
class BufReader {
public:
    BufReader(const uint8_t* data, size_t size)
        : data_(data), size_(size), pos_(0) {}

    bool has_bytes(size_t n) const { return pos_ + n <= size_; }
    size_t pos() const { return pos_; }

    uint8_t  read_u8()  { uint8_t  v; memcpy(&v, data_ + pos_, 1); pos_ += 1; return v; }
    uint16_t read_u16() { uint16_t v; memcpy(&v, data_ + pos_, 2); pos_ += 2; return v; }
    uint32_t read_u32() { uint32_t v; memcpy(&v, data_ + pos_, 4); pos_ += 4; return v; }
    uint64_t read_u64() { uint64_t v; memcpy(&v, data_ + pos_, 8); pos_ += 8; return v; }
    int32_t  read_i32() { int32_t  v; memcpy(&v, data_ + pos_, 4); pos_ += 4; return v; }
    float    read_f32() { float    v; memcpy(&v, data_ + pos_, 4); pos_ += 4; return v; }

    std::string read_string() {
        uint64_t len = read_u64();
        if (pos_ + len > size_) return "";
        std::string s(reinterpret_cast<const char*>(data_ + pos_), len);
        pos_ += len;
        return s;
    }

    // Skip a GGUF metadata value (for values we don't need)
    void skip_value(uint32_t type) {
        switch (type) {
            case GGUF_TYPE_UINT8:
            case GGUF_TYPE_INT8:
            case GGUF_TYPE_BOOL:    pos_ += 1; break;
            case GGUF_TYPE_UINT16:
            case GGUF_TYPE_INT16:   pos_ += 2; break;
            case GGUF_TYPE_UINT32:
            case GGUF_TYPE_INT32:
            case GGUF_TYPE_FLOAT32: pos_ += 4; break;
            case GGUF_TYPE_UINT64:
            case GGUF_TYPE_INT64:
            case GGUF_TYPE_FLOAT64: pos_ += 8; break;
            case GGUF_TYPE_STRING:  read_string(); break;
            case GGUF_TYPE_ARRAY: {
                uint32_t atype = read_u32();
                uint64_t alen  = read_u64();
                for (uint64_t i = 0; i < alen; ++i) skip_value(atype);
                break;
            }
        }
    }

    uint64_t read_value_u64(uint32_t type) {
        switch (type) {
            case GGUF_TYPE_UINT8:   return read_u8();
            case GGUF_TYPE_UINT16:  return read_u16();
            case GGUF_TYPE_UINT32:  return read_u32();
            case GGUF_TYPE_UINT64:  return read_u64();
            case GGUF_TYPE_INT32:   return static_cast<uint64_t>(read_i32());
            default: skip_value(type); return 0;
        }
    }

    std::string read_value_string(uint32_t type) {
        if (type == GGUF_TYPE_STRING) return read_string();
        skip_value(type);
        return "";
    }

private:
    const uint8_t* data_;
    size_t size_;
    size_t pos_;
};

bool ExpertMap::load(const std::string& gguf_path) {
    gguf_path_ = gguf_path;

    // Read the full file into memory (for metadata parsing only)
    // We'll mmap later for actual weight access
    std::ifstream file(gguf_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        fprintf(stderr, "[ExpertMap] Failed to open: %s\n", gguf_path.c_str());
        return false;
    }

    size_t file_size = file.tellg();
    file.seekg(0);

    // Only read the header + metadata (first ~1MB is enough for most models)
    // For tensor info, we need to read more but not the full data section
    size_t header_read_size = std::min(file_size, static_cast<size_t>(256 * 1024 * 1024));
    std::vector<uint8_t> header_buf(header_read_size);
    file.read(reinterpret_cast<char*>(header_buf.data()), header_read_size);
    file.close();

    if (!parse_metadata(header_buf.data(), header_read_size)) {
        return false;
    }

    if (!classify_tensors(header_buf.data(), header_read_size)) {
        return false;
    }

    return true;
}

bool ExpertMap::parse_metadata(const uint8_t* data, size_t file_size) {
    BufReader r(data, file_size);

    // Check magic
    uint32_t magic = r.read_u32();
    if (magic != GGUF_MAGIC) {
        fprintf(stderr, "[ExpertMap] Not a GGUF file (magic: 0x%08X)\n", magic);
        return false;
    }

    uint32_t version = r.read_u32();
    if (version < 2 || version > 3) {
        fprintf(stderr, "[ExpertMap] Unsupported GGUF version: %u\n", version);
        return false;
    }

    uint64_t n_tensors = r.read_u64();
    uint64_t n_kv      = r.read_u64();

    // Parse metadata key-value pairs
    for (uint64_t i = 0; i < n_kv; ++i) {
        if (!r.has_bytes(8)) break;

        std::string key = r.read_string();
        uint32_t vtype = r.read_u32();

        // Extract architecture-specific metadata
        if (key.find(".block_count") != std::string::npos) {
            arch_.n_layers = static_cast<uint32_t>(r.read_value_u64(vtype));
        } else if (key.find(".expert_count") != std::string::npos &&
                   key.find("used") == std::string::npos) {
            arch_.n_experts = static_cast<uint32_t>(r.read_value_u64(vtype));
        } else if (key.find(".expert_used_count") != std::string::npos) {
            arch_.n_experts_used = static_cast<uint32_t>(r.read_value_u64(vtype));
        } else if (key.find(".embedding_length") != std::string::npos) {
            arch_.embed_dim = static_cast<uint32_t>(r.read_value_u64(vtype));
        } else if (key.find(".expert_feed_forward_length") != std::string::npos) {
            arch_.expert_ffn_dim = static_cast<uint32_t>(r.read_value_u64(vtype));
        } else if (key.find(".feed_forward_length") != std::string::npos &&
                   key.find("expert") == std::string::npos &&
                   key.find("shared") == std::string::npos) {
            // General FFN dim — use as fallback if expert-specific key is absent
            // (e.g. Mixtral uses this for all experts)
            if (arch_.expert_ffn_dim == 0) {
                arch_.expert_ffn_dim = static_cast<uint32_t>(r.read_value_u64(vtype));
            }
        } else if (key.find(".expert_shared_feed_forward_length") != std::string::npos) {
            arch_.shared_expert_ffn_dim = static_cast<uint32_t>(r.read_value_u64(vtype));
        } else if (key.find(".attention.head_count") != std::string::npos &&
                   key.find("kv") == std::string::npos) {
            arch_.n_heads = static_cast<uint32_t>(r.read_value_u64(vtype));
        } else if (key.find(".attention.head_count_kv") != std::string::npos) {
            arch_.n_kv_heads = static_cast<uint32_t>(r.read_value_u64(vtype));
        } else if (key.find(".context_length") != std::string::npos) {
            arch_.context_length = r.read_value_u64(vtype);
        } else if (key == "general.architecture") {
            arch_.architecture = r.read_value_string(vtype);
        } else if (key == "general.name") {
            arch_.model_name = r.read_value_string(vtype);
        } else {
            r.skip_value(vtype);
        }
    }

    // Parse tensor info to build tensor locations
    // Each tensor info: name (string) + n_dims (u32) + ne[n_dims] (u64[]) + type (u32) + offset (u64)
    // (tensor info follows metadata)

    struct RawTensorInfo {
        std::string name;
        uint32_t    n_dims;
        uint64_t    ne[4];
        uint32_t    type;
        uint64_t    offset;
    };

    std::vector<RawTensorInfo> raw_tensors;
    raw_tensors.reserve(n_tensors);

    for (uint64_t i = 0; i < n_tensors; ++i) {
        if (!r.has_bytes(16)) break;

        RawTensorInfo ti{};
        ti.name = r.read_string();
        ti.n_dims = r.read_u32();
        for (uint32_t d = 0; d < ti.n_dims; ++d) {
            ti.ne[d] = r.read_u64();
        }
        for (uint32_t d = ti.n_dims; d < 4; ++d) {
            ti.ne[d] = 1;
        }
        ti.type = r.read_u32();
        ti.offset = r.read_u64();
        raw_tensors.push_back(std::move(ti));
    }

    // Data section starts after all tensor info, aligned to 32 bytes
    data_offset_ = (r.pos() + 31) & ~static_cast<size_t>(31);

    // Store raw tensor info for classification
    // Compute actual byte sizes for each tensor
    for (auto& ti : raw_tensors) {
        TensorLocation loc;
        loc.name       = ti.name;
        loc.offset     = data_offset_ + ti.offset;
        loc.n_dims     = ti.n_dims;
        loc.quant_type = ti.type;
        for (uint32_t d = 0; d < 4; ++d) loc.ne[d] = ti.ne[d];

        // Compute tensor size in bytes: (n_elements / blck_size) * type_size
        uint64_t n_elements = 1;
        for (uint32_t d = 0; d < ti.n_dims; ++d) n_elements *= ti.ne[d];

        auto tinfo = ggml_type_info(ti.type);
        if (tinfo.blck_size > 1) {
            loc.size_bytes = (n_elements / tinfo.blck_size) * tinfo.type_size;
        } else {
            loc.size_bytes = n_elements * tinfo.type_size;
        }

        // Temporarily store all tensors in shared_ — we'll reclassify next
        shared_.push_back(std::move(loc));
    }

    return true;
}

bool ExpertMap::classify_tensors(const uint8_t* /*data*/, size_t /*file_size*/) {
    if (arch_.n_experts == 0) {
        // Not a MoE model — all tensors are "shared"
        return true;
    }

    // Regex patterns for expert tensors in GGUF
    //
    // Pattern A (stacked): blk.{L}.ffn_gate_exps — shape [n_experts, rows, cols]
    //   Used by: Qwen MoE, DeepSeek V2/V3, Llama 4, DBRX, Grok
    // Pattern B (per-expert): blk.{L}.ffn_gate.{E} — shape [rows, cols]
    //   Used by: Mixtral 8x7B/8x22B
    // Pattern C (fused): blk.{L}.ffn_gate_up_exps — gate+up merged
    //   Used by: Some Qwen variants, DBRX
    //
    // Pattern A: stacked expert tensors
    std::regex expert_gate_re(R"(blk\.(\d+)\.ffn_gate_exps)");
    std::regex expert_up_re(R"(blk\.(\d+)\.ffn_up_exps)");
    std::regex expert_down_re(R"(blk\.(\d+)\.ffn_down_exps)");
    // Pattern B: per-expert tensors (Mixtral-style)
    std::regex expert_gate_per_re(R"(blk\.(\d+)\.ffn_gate\.(\d+))");
    std::regex expert_up_per_re(R"(blk\.(\d+)\.ffn_up\.(\d+))");
    std::regex expert_down_per_re(R"(blk\.(\d+)\.ffn_down\.(\d+))");
    // Pattern C: fused gate_up
    std::regex expert_gate_up_re(R"(blk\.(\d+)\.ffn_gate_up_exps)");

    // Separate shared from expert tensors
    std::vector<TensorLocation> new_shared;
    new_shared.reserve(shared_.size());

    for (auto& loc : shared_) {
        std::smatch match;

        auto try_decompose = [&](const std::regex& re, ExpertProj proj) -> bool {
            if (!std::regex_search(loc.name, match, re)) return false;

            uint32_t layer_id = static_cast<uint32_t>(std::stoi(match[1].str()));

            // Merged tensor shape: [n_experts, rows, cols] (in ggml order: ne[0]=cols, ne[1]=rows, ne[2]=n_experts)
            uint32_t n_exp = (loc.n_dims >= 3) ? static_cast<uint32_t>(loc.ne[2]) : 1;
            uint64_t rows  = loc.ne[1];
            uint64_t cols  = loc.ne[0];
            uint64_t per_expert_bytes = loc.size_bytes / n_exp;

            for (uint32_t eid = 0; eid < n_exp; ++eid) {
                ExpertSlice slice;
                slice.layer_id   = layer_id;
                slice.expert_id  = eid;
                slice.proj       = proj;
                slice.offset     = loc.offset + eid * per_expert_bytes;
                slice.size_bytes = per_expert_bytes;
                slice.rows       = rows;
                slice.cols       = cols;
                slice.quant_type = loc.quant_type;

                ExpertKey key{layer_id, eid, proj};
                expert_index_[key] = experts_.size();
                experts_.push_back(slice);
            }
            return true;
        };

        if (try_decompose(expert_gate_re, ExpertProj::kGateProj)) continue;
        if (try_decompose(expert_up_re, ExpertProj::kUpProj)) continue;
        if (try_decompose(expert_down_re, ExpertProj::kDownProj)) continue;

        // Pattern B: per-expert tensors (Mixtral-style: blk.L.ffn_gate.E)
        auto try_per_expert = [&](const std::regex& re, ExpertProj proj) -> bool {
            if (!std::regex_search(loc.name, match, re)) return false;

            uint32_t layer_id  = static_cast<uint32_t>(std::stoi(match[1].str()));
            uint32_t expert_id = static_cast<uint32_t>(std::stoi(match[2].str()));

            ExpertSlice slice;
            slice.layer_id   = layer_id;
            slice.expert_id  = expert_id;
            slice.proj       = proj;
            slice.offset     = loc.offset;
            slice.size_bytes = loc.size_bytes;
            slice.rows       = loc.ne[1];
            slice.cols       = loc.ne[0];
            slice.quant_type = loc.quant_type;

            ExpertKey key{layer_id, expert_id, proj};
            expert_index_[key] = experts_.size();
            experts_.push_back(slice);
            return true;
        };

        if (try_per_expert(expert_gate_per_re, ExpertProj::kGateProj)) continue;
        if (try_per_expert(expert_up_per_re, ExpertProj::kUpProj)) continue;
        if (try_per_expert(expert_down_per_re, ExpertProj::kDownProj)) continue;

        // Handle fused gate_up tensor: decompose into gate + up halves
        if (std::regex_search(loc.name, match, expert_gate_up_re)) {
            uint32_t layer_id = static_cast<uint32_t>(std::stoi(match[1].str()));
            uint32_t n_exp = (loc.n_dims >= 3) ? static_cast<uint32_t>(loc.ne[2]) : 1;
            uint64_t rows  = loc.ne[1];
            uint64_t cols  = loc.ne[0];
            uint64_t per_expert_bytes = loc.size_bytes / n_exp;
            uint64_t half_bytes = per_expert_bytes / 2;

            for (uint32_t eid = 0; eid < n_exp; ++eid) {
                // Gate half
                ExpertSlice gate_slice;
                gate_slice.layer_id   = layer_id;
                gate_slice.expert_id  = eid;
                gate_slice.proj       = ExpertProj::kGateProj;
                gate_slice.offset     = loc.offset + eid * per_expert_bytes;
                gate_slice.size_bytes = half_bytes;
                gate_slice.rows       = rows / 2;
                gate_slice.cols       = cols;
                gate_slice.quant_type = loc.quant_type;

                ExpertKey gate_key{layer_id, eid, ExpertProj::kGateProj};
                expert_index_[gate_key] = experts_.size();
                experts_.push_back(gate_slice);

                // Up half
                ExpertSlice up_slice;
                up_slice.layer_id   = layer_id;
                up_slice.expert_id  = eid;
                up_slice.proj       = ExpertProj::kUpProj;
                up_slice.offset     = loc.offset + eid * per_expert_bytes + half_bytes;
                up_slice.size_bytes = half_bytes;
                up_slice.rows       = rows / 2;
                up_slice.cols       = cols;
                up_slice.quant_type = loc.quant_type;

                ExpertKey up_key{layer_id, eid, ExpertProj::kUpProj};
                expert_index_[up_key] = experts_.size();
                experts_.push_back(up_slice);
            }
            continue;
        }

        // Not an expert tensor — keep as shared
        new_shared.push_back(std::move(loc));
    }

    shared_ = std::move(new_shared);

    // If expert_ffn_dim was not set from expert-specific key, try general FFN dim
    if (arch_.expert_ffn_dim == 0 && arch_.n_experts > 0) {
        // Infer from first gate expert tensor if available
        for (const auto& e : experts_) {
            if (e.proj == ExpertProj::kGateProj) {
                arch_.expert_ffn_dim = static_cast<uint32_t>(e.rows);
                break;
            }
        }
    }

    // Infer n_experts from actual tensor data if metadata didn't set it
    if (arch_.n_experts == 0 && !experts_.empty()) {
        uint32_t max_eid = 0;
        for (const auto& e : experts_) {
            max_eid = std::max(max_eid, e.expert_id);
        }
        arch_.n_experts = max_eid + 1;
    }

    // Compute architecture summary
    if (!experts_.empty()) {
        uint64_t total_expert = 0;
        uint64_t per_expert_total = 0;
        for (const auto& e : experts_) total_expert += e.size_bytes;

        // Per-expert size (all 3 projections)
        uint32_t projs_per_expert = static_cast<uint32_t>(ExpertProj::kCount);
        if (arch_.n_experts > 0 && arch_.n_layers > 0 && projs_per_expert > 0) {
            per_expert_total = total_expert / (arch_.n_layers * arch_.n_experts);
        }

        uint64_t total_shared = 0;
        for (const auto& s : shared_) total_shared += s.size_bytes;

        arch_.expert_weight_bytes    = per_expert_total;
        arch_.total_expert_bytes     = total_expert;
        arch_.shared_weight_bytes    = total_shared;
        // Per-layer: shared_per_layer + top-K experts
        arch_.active_bytes_per_layer = total_shared / arch_.n_layers +
            arch_.n_experts_used * per_expert_total;
        // Per-token: sum across all layers (this is what determines tok/s)
        arch_.active_bytes_per_token = arch_.active_bytes_per_layer * arch_.n_layers;
    }

    return true;
}

const ExpertSlice* ExpertMap::get_expert(uint32_t layer_id, uint32_t expert_id,
                                          ExpertProj proj) const {
    ExpertKey key{layer_id, expert_id, proj};
    auto it = expert_index_.find(key);
    if (it == expert_index_.end()) return nullptr;
    return &experts_[it->second];
}

std::vector<const ExpertSlice*> ExpertMap::get_layer_experts(uint32_t layer_id) const {
    std::vector<const ExpertSlice*> result;
    for (const auto& e : experts_) {
        if (e.layer_id == layer_id) {
            result.push_back(&e);
        }
    }
    return result;
}

std::vector<const ExpertSlice*> ExpertMap::get_active_experts(
    uint32_t layer_id, const std::vector<uint32_t>& expert_ids) const {
    std::vector<const ExpertSlice*> result;
    result.reserve(expert_ids.size() * static_cast<uint32_t>(ExpertProj::kCount));

    for (uint32_t eid : expert_ids) {
        for (uint32_t p = 0; p < static_cast<uint32_t>(ExpertProj::kCount); ++p) {
            const ExpertSlice* s = get_expert(layer_id, eid, static_cast<ExpertProj>(p));
            if (s) result.push_back(s);
        }
    }
    return result;
}

void ExpertMap::print_summary() const {
    printf("=== ExpertMap: %s ===\n", arch_.model_name.c_str());
    printf("Architecture: %s\n", arch_.architecture.c_str());
    printf("Layers: %u, Experts/layer: %u, Top-K: %u\n",
           arch_.n_layers, arch_.n_experts, arch_.n_experts_used);
    printf("Embed dim: %u, Expert FFN: %u, Shared FFN: %u\n",
           arch_.embed_dim, arch_.expert_ffn_dim, arch_.shared_expert_ffn_dim);
    printf("Attention: %u heads, %u KV heads (GQA)\n",
           arch_.n_heads, arch_.n_kv_heads);
    printf("Context: %lu tokens\n", arch_.context_length);
    printf("\n");
    printf("Shared tensors: %zu (%.1f MB)\n",
           shared_.size(), arch_.shared_weight_bytes / (1024.0 * 1024.0));
    printf("Expert slices:  %zu (%.1f GB)\n",
           experts_.size(), arch_.total_expert_bytes / (1024.0 * 1024.0 * 1024.0));
    printf("Per expert:     %.2f MB (3 projections)\n",
           arch_.expert_weight_bytes / (1024.0 * 1024.0));
    printf("Active/layer:   %.1f MB (shared %.1f + %u experts × %.2f MB)\n",
           arch_.active_bytes_per_layer / (1024.0 * 1024.0),
           (arch_.shared_weight_bytes / arch_.n_layers) / (1024.0 * 1024.0),
           arch_.n_experts_used,
           arch_.expert_weight_bytes / (1024.0 * 1024.0));
    printf("Active/token:   %.1f MB (%u layers × %.1f MB/layer)\n",
           arch_.active_bytes_per_token / (1024.0 * 1024.0),
           arch_.n_layers,
           arch_.active_bytes_per_layer / (1024.0 * 1024.0));
    printf("\n");
    printf("Speed estimates (generation tok/s):\n");
    printf("  GPU VRAM (288 GB/s): %.0f tok/s\n", arch_.estimate_speed(288e9));
    printf("  DDR4 RAM  (40 GB/s): %.0f tok/s\n", arch_.estimate_speed(40e9));
    printf("  PCIe 4.0  (25 GB/s): %.0f tok/s\n", arch_.estimate_speed(25e9));
    if (arch_.n_experts > 0) {
        // Full layer = all experts + shared; active layer = top-K experts + shared
        double full_layer_bytes = static_cast<double>(arch_.total_expert_bytes) / arch_.n_layers +
                                  static_cast<double>(arch_.shared_weight_bytes) / arch_.n_layers;
        double active_layer_bytes = static_cast<double>(arch_.active_bytes_per_layer);
        double waste_pct = 100.0 * (1.0 - active_layer_bytes / full_layer_bytes);
        printf("  Without ExpertFlow: %.0f%% bandwidth wasted on inactive experts\n", waste_pct);
    }
    printf("=== End ExpertMap ===\n");
}

}  // namespace expertflow
