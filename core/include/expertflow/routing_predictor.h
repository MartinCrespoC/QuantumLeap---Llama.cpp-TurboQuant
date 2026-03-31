// ExpertFlow — Adaptive Routing Predictor
// routing_predictor.h: Markov chain expert prediction for prefetching
//
// Tracks expert co-activation patterns across layers and tokens to predict
// which experts will be needed next. Improves prefetch accuracy from ~80%
// (naive layer-repeat) to ~90-95% (learned patterns).
//
// Strategy:
//   1. Maintain per-layer frequency counts: freq[layer][expert] (popularity)
//   2. Maintain cross-layer transitions: trans[layer][expert_a] → expert_b
//   3. Combine popularity + transition for next-layer prediction
//   4. Exponential moving average with decay (adapts to context shift)
//
// Memory: O(n_layers × n_experts) for frequency + O(n_layers × top_k²) for transitions

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace expertflow {

// Prediction result: ordered list of predicted experts with confidence
struct PredictedExperts {
    std::vector<uint32_t> expert_ids;    // Predicted expert IDs (best first)
    std::vector<float>    confidences;   // Confidence scores [0, 1]
};

// Configuration for the routing predictor
struct PredictorConfig {
    uint32_t n_layers;           // Total transformer layers
    uint32_t n_experts;          // Experts per layer
    uint32_t n_experts_used;     // Top-K experts per token (for sizing)
    float    decay;              // EMA decay factor (0.9 = slow adapt, 0.5 = fast)
    float    transition_weight;  // How much to weight transitions vs popularity [0,1]
    uint32_t predict_count;      // How many experts to predict (>= n_experts_used)

    static PredictorConfig defaults(uint32_t layers, uint32_t experts,
                                     uint32_t top_k) {
        return PredictorConfig{
            .n_layers          = layers,
            .n_experts         = experts,
            .n_experts_used    = top_k,
            .decay             = 0.85f,
            .transition_weight = 0.6f,
            .predict_count     = top_k + top_k / 2,  // 50% speculative
        };
    }
};

// Prediction accuracy statistics
struct PredictorStats {
    uint64_t predictions;        // Total predictions made
    uint64_t correct;            // Predictions that were actually used
    uint64_t total_experts;      // Total experts actually routed
    double   accuracy;           // correct / total_experts

    void update(uint32_t predicted_correct, uint32_t actual_count) {
        predictions++;
        correct += predicted_correct;
        total_experts += actual_count;
        accuracy = total_experts > 0
            ? static_cast<double>(correct) / total_experts
            : 0.0;
    }
};

// Adaptive Routing Predictor
//
// Usage:
//   1. Create with config
//   2. For each token, for each layer:
//      a. Call predict(layer) to get predicted experts for that layer
//      b. Call observe(layer, actual_experts) after routing is known
//   3. Call begin_token() between tokens to advance the EMA
//
class AdaptiveRoutingPredictor {
public:
    AdaptiveRoutingPredictor() = default;

    // Initialize predictor with model dimensions
    void init(const PredictorConfig& config);

    // Reset all learned patterns
    void reset();

    // Begin a new token (applies EMA decay to frequencies)
    void begin_token();

    // Observe the actual routing result for a layer.
    // Updates frequency counts and transition matrix.
    void observe(uint32_t layer_id,
                 const std::vector<uint32_t>& expert_ids);

    // Predict the next layer's experts based on current layer's routing.
    // Returns the top predict_count experts sorted by predicted likelihood.
    //   current_layer: the layer that just finished routing
    //   current_experts: expert IDs selected at current_layer
    PredictedExperts predict(uint32_t current_layer,
                             const std::vector<uint32_t>& current_experts) const;

    // Predict experts for a layer using only popularity (no transition info).
    // Useful for the first layer of each token where no transition data exists.
    PredictedExperts predict_popular(uint32_t layer_id) const;

    // Statistics
    const PredictorStats& stats() const { return stats_; }

    // Score a prediction against actual routing (updates stats)
    void score_prediction(const PredictedExperts& predicted,
                          const std::vector<uint32_t>& actual);

    // Is the predictor initialized?
    bool is_initialized() const { return initialized_; }

private:
    bool initialized_ = false;
    PredictorConfig config_{};
    PredictorStats stats_{};

    // Per-layer expert frequency: freq_[layer][expert] = EMA frequency score
    std::vector<std::vector<float>> freq_;

    // Cross-layer transition counts:
    // trans_[layer][i] = frequency of expert i at layer being followed by
    // each expert at layer+1. Stored as flat: trans_[layer][expert * n_experts + next_expert]
    // To save memory, only track transitions for the top-K most popular experts.
    // Format: trans_[layer] is a vector of (from_expert, to_expert, count) tuples.
    struct Transition {
        uint32_t from_expert;
        uint32_t to_expert;
        float    count;
    };
    std::vector<std::vector<Transition>> trans_;

    // Last layer's routing (for building transitions on next observe)
    uint32_t last_layer_ = UINT32_MAX;
    std::vector<uint32_t> last_experts_;

    // Token counter for EMA
    uint64_t token_count_ = 0;
};

}  // namespace expertflow
