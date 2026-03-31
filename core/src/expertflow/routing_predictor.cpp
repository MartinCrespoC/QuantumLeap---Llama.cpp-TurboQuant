// ExpertFlow — routing_predictor.cpp
// Adaptive Markov chain routing predictor for expert prefetching
//
// Learns expert co-activation patterns across layers:
//   1. Popularity: EMA of per-layer expert frequency
//   2. Transitions: which experts at layer L predict which at layer L+1
//   3. Combined score: transition_weight × transition + (1-tw) × popularity

#include "expertflow/routing_predictor.h"

#include <algorithm>
#include <cstring>

namespace expertflow {

// ============================================================
// Initialization
// ============================================================

void AdaptiveRoutingPredictor::init(const PredictorConfig& config) {
    config_ = config;
    stats_ = {};
    token_count_ = 0;
    last_layer_ = UINT32_MAX;
    last_experts_.clear();

    // Initialize frequency table: [n_layers][n_experts] = 0
    freq_.resize(config.n_layers);
    for (auto& layer_freq : freq_) {
        layer_freq.assign(config.n_experts, 0.0f);
    }

    // Initialize transition table: [n_layers] = empty
    trans_.resize(config.n_layers);

    initialized_ = true;
}

void AdaptiveRoutingPredictor::reset() {
    if (!initialized_) return;
    stats_ = {};
    token_count_ = 0;
    last_layer_ = UINT32_MAX;
    last_experts_.clear();

    for (auto& layer_freq : freq_) {
        std::fill(layer_freq.begin(), layer_freq.end(), 0.0f);
    }
    for (auto& layer_trans : trans_) {
        layer_trans.clear();
    }
}

// ============================================================
// Token lifecycle
// ============================================================

void AdaptiveRoutingPredictor::begin_token() {
    token_count_++;
    last_layer_ = UINT32_MAX;
    last_experts_.clear();

    // Apply EMA decay to frequency counts
    float decay = config_.decay;
    for (auto& layer_freq : freq_) {
        for (float& f : layer_freq) {
            f *= decay;
        }
    }

    // Decay transition counts
    for (auto& layer_trans : trans_) {
        for (auto& t : layer_trans) {
            t.count *= decay;
        }
        // Prune very small transitions to save memory
        layer_trans.erase(
            std::remove_if(layer_trans.begin(), layer_trans.end(),
                           [](const Transition& t) { return t.count < 0.01f; }),
            layer_trans.end());
    }
}

// ============================================================
// Observe routing result
// ============================================================

void AdaptiveRoutingPredictor::observe(uint32_t layer_id,
                                const std::vector<uint32_t>& expert_ids) {
    if (!initialized_ || layer_id >= config_.n_layers) return;

    // Update frequency: increment selected experts
    for (uint32_t eid : expert_ids) {
        if (eid < config_.n_experts) {
            freq_[layer_id][eid] += 1.0f;
        }
    }

    // Update transitions from previous layer
    if (last_layer_ != UINT32_MAX && last_layer_ + 1 == layer_id) {
        auto& layer_trans = trans_[last_layer_];

        for (uint32_t from : last_experts_) {
            for (uint32_t to : expert_ids) {
                // Find existing transition or create new
                bool found = false;
                for (auto& t : layer_trans) {
                    if (t.from_expert == from && t.to_expert == to) {
                        t.count += 1.0f;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    layer_trans.push_back({from, to, 1.0f});
                }
            }
        }
    }

    last_layer_ = layer_id;
    last_experts_ = expert_ids;
}

// ============================================================
// Predict next layer's experts
// ============================================================

PredictedExperts AdaptiveRoutingPredictor::predict(
    uint32_t current_layer,
    const std::vector<uint32_t>& current_experts) const {

    PredictedExperts result;
    if (!initialized_) return result;

    uint32_t next_layer = current_layer + 1;
    if (next_layer >= config_.n_layers) {
        return predict_popular(current_layer);
    }

    // Build score for each expert at next_layer
    std::vector<float> scores(config_.n_experts, 0.0f);

    // Component 1: Popularity at next layer
    float pop_weight = 1.0f - config_.transition_weight;
    if (next_layer < freq_.size()) {
        float max_freq = 0.0f;
        for (float f : freq_[next_layer]) {
            max_freq = std::max(max_freq, f);
        }
        if (max_freq > 0.0f) {
            for (uint32_t e = 0; e < config_.n_experts; ++e) {
                scores[e] += pop_weight * (freq_[next_layer][e] / max_freq);
            }
        }
    }

    // Component 2: Transition probability from current experts
    float trans_weight = config_.transition_weight;
    if (current_layer < trans_.size()) {
        const auto& layer_trans = trans_[current_layer];

        // Sum transition counts to each next expert from current experts
        for (const auto& t : layer_trans) {
            // Check if from_expert is in current_experts
            for (uint32_t ce : current_experts) {
                if (t.from_expert == ce && t.to_expert < config_.n_experts) {
                    scores[t.to_expert] += trans_weight * t.count;
                }
            }
        }
    }

    // Sort by score, take top predict_count
    std::vector<std::pair<float, uint32_t>> scored;
    scored.reserve(config_.n_experts);
    for (uint32_t e = 0; e < config_.n_experts; ++e) {
        if (scores[e] > 0.0f) {
            scored.emplace_back(scores[e], e);
        }
    }

    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    uint32_t count = std::min(config_.predict_count,
                               static_cast<uint32_t>(scored.size()));

    // Normalize confidences
    float max_score = count > 0 ? scored[0].first : 1.0f;
    if (max_score <= 0.0f) max_score = 1.0f;

    result.expert_ids.resize(count);
    result.confidences.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
        result.expert_ids[i] = scored[i].second;
        result.confidences[i] = scored[i].first / max_score;
    }

    // If we don't have enough predictions, fill with popular experts
    if (count < config_.predict_count) {
        auto popular = predict_popular(next_layer);
        for (uint32_t i = 0; i < popular.expert_ids.size() &&
             result.expert_ids.size() < config_.predict_count; ++i) {
            uint32_t eid = popular.expert_ids[i];
            // Don't duplicate
            bool already = false;
            for (uint32_t existing : result.expert_ids) {
                if (existing == eid) { already = true; break; }
            }
            if (!already) {
                result.expert_ids.push_back(eid);
                result.confidences.push_back(popular.confidences[i] * 0.5f);
            }
        }
    }

    return result;
}

PredictedExperts AdaptiveRoutingPredictor::predict_popular(uint32_t layer_id) const {
    PredictedExperts result;
    if (!initialized_ || layer_id >= config_.n_layers) return result;

    const auto& lf = freq_[layer_id];
    std::vector<std::pair<float, uint32_t>> scored;
    scored.reserve(config_.n_experts);

    for (uint32_t e = 0; e < config_.n_experts; ++e) {
        if (lf[e] > 0.0f) {
            scored.emplace_back(lf[e], e);
        }
    }

    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    uint32_t count = std::min(config_.predict_count,
                               static_cast<uint32_t>(scored.size()));

    float max_score = count > 0 ? scored[0].first : 1.0f;
    if (max_score <= 0.0f) max_score = 1.0f;

    result.expert_ids.resize(count);
    result.confidences.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
        result.expert_ids[i] = scored[i].second;
        result.confidences[i] = scored[i].first / max_score;
    }

    return result;
}

// ============================================================
// Score prediction accuracy
// ============================================================

void AdaptiveRoutingPredictor::score_prediction(
    const PredictedExperts& predicted,
    const std::vector<uint32_t>& actual) {

    uint32_t correct = 0;
    for (uint32_t a : actual) {
        for (uint32_t p : predicted.expert_ids) {
            if (p == a) { ++correct; break; }
        }
    }
    stats_.update(correct, static_cast<uint32_t>(actual.size()));
}

}  // namespace expertflow
