// ExpertFlow — expertflow_backend.cpp
// ggml Backend Integration Bridge implementation
//
// Provides tensor classification, global session management,
// and the runtime hooks for integrating with llama.cpp's inference loop.

#include "expertflow/expertflow_backend.h"

#include <cstdio>
#include <mutex>
#include <regex>

namespace expertflow {

// ============================================================
// Tensor Classification
// ============================================================

TensorPlacement classify_tensor(const std::string& name) {
    if (is_expert_tensor(name)) {
        return TensorPlacement::kCPU;
    }
    if (is_shared_tensor(name)) {
        return TensorPlacement::kGPU;
    }
    return TensorPlacement::kDefault;
}

bool is_expert_tensor(const std::string& name) {
    // Expert projection tensors follow multiple naming patterns in GGUF:
    //
    // Pattern A (stacked, Qwen/DeepSeek/Llama4/DBRX/Grok):
    //   blk.{L}.ffn_gate_exps.weight   — all experts stacked [n_exp, rows, cols]
    //   blk.{L}.ffn_up_exps.weight
    //   blk.{L}.ffn_down_exps.weight
    //   blk.{L}.ffn_gate_up_exps.weight — fused gate+up
    //
    // Pattern B (per-expert, Mixtral 8x7B/8x22B):
    //   blk.{L}.ffn_gate.{E}.weight    — individual expert tensors
    //   blk.{L}.ffn_up.{E}.weight
    //   blk.{L}.ffn_down.{E}.weight

    // Pattern A: stacked expert tensors (fast string match)
    if (name.find("ffn_gate_exps") != std::string::npos ||
        name.find("ffn_up_exps")   != std::string::npos ||
        name.find("ffn_down_exps") != std::string::npos ||
        name.find("ffn_gate_up_exps") != std::string::npos) {
        return true;
    }

    // Pattern B: per-expert tensors (Mixtral-style: blk.L.ffn_gate.E)
    // Match ffn_gate.{digit} / ffn_up.{digit} / ffn_down.{digit}
    // but NOT ffn_gate_inp or ffn_gate_exps (already matched above)
    static const std::regex per_expert_re(
        R"(blk\.\d+\.ffn_(?:gate|up|down)\.\d+)");
    if (std::regex_search(name, per_expert_re)) {
        return true;
    }

    return false;
}

bool is_shared_tensor(const std::string& name) {
    // Everything that is NOT an expert tensor is shared:
    // - Attention: attn_q, attn_k, attn_v, attn_output
    // - Norms: attn_norm, ffn_norm, output_norm
    // - Router: ffn_gate_inp (routes tokens to experts)
    // - Shared expert: ffn_gate, ffn_up, ffn_down (without _exps or .{E} suffix)
    // - Embeddings: token_embd, output
    // - SSM/Mamba: ssm_*
    // - MLA: attn_kv_a, attn_kv_b (DeepSeek MLA)
    return !is_expert_tensor(name);
}

// ============================================================
// ExpertFlowBackend
// ============================================================

ExpertFlowBackend::~ExpertFlowBackend() {
    pipeline_.release();
}

std::unique_ptr<ExpertFlowBackend> ExpertFlowBackend::create(
    const std::string& gguf_path,
    const BackendConfig& config) {

    auto backend = std::unique_ptr<ExpertFlowBackend>(new ExpertFlowBackend());
    backend->config_ = config;

    if (!config.enabled) {
        printf("[ExpertFlow] Disabled by config\n");
        return nullptr;
    }

    // Build pipeline config from backend config
    PipelineConfig pcfg;
    pcfg.expert_cache_vram_bytes = config.expert_cache_bytes;
    pcfg.staging_buffer_bytes    = config.staging_buffer_bytes;
    pcfg.enable_prefetch         = config.enable_prefetch;
    pcfg.enable_coalescing       = config.enable_coalescing;
    pcfg.enable_profiling        = config.enable_profiling;
    pcfg.speculative_top_k       = config.speculative_top_k;
    pcfg.reserved_hot_per_layer  = config.reserved_hot_per_layer;
    pcfg.recency_weight          = config.recency_weight;

    if (!backend->pipeline_.init(gguf_path, pcfg)) {
        fprintf(stderr, "[ExpertFlow] Failed to initialize pipeline\n");
        return nullptr;
    }

    const auto& arch = backend->pipeline_.expert_map().architecture();
    printf("[ExpertFlow] Backend active for %s (%u layers, %u experts, top-%u)\n",
           arch.model_name.c_str(), arch.n_layers, arch.n_experts, arch.n_experts_used);

    return backend;
}

void ExpertFlowBackend::begin_token() {
    if (pipeline_.is_ready()) {
        pipeline_.begin_token();
    }
}

PipelineController::ExpertPointers ExpertFlowBackend::prepare_experts(
    uint32_t layer_id,
    const std::vector<uint32_t>& expert_ids,
    const std::vector<float>& gate_weights) {

    LayerRouting routing;
    routing.layer_id    = layer_id;
    routing.expert_ids  = expert_ids;
    routing.gate_weights = gate_weights;

    return pipeline_.process_layer(routing);
}

void ExpertFlowBackend::end_token() {
    if (pipeline_.is_ready()) {
        pipeline_.end_token();
    }
}

// ============================================================
// Global instance management
// ============================================================

static std::unique_ptr<ExpertFlowBackend> g_backend;
static std::mutex g_backend_mutex;

void set_global_backend(std::unique_ptr<ExpertFlowBackend> backend) {
    std::lock_guard<std::mutex> lock(g_backend_mutex);
    g_backend = std::move(backend);
}

ExpertFlowBackend* get_global_backend() {
    std::lock_guard<std::mutex> lock(g_backend_mutex);
    return g_backend.get();
}

bool is_expertflow_active() {
    auto* backend = get_global_backend();
    return backend && backend->is_active();
}

}  // namespace expertflow
