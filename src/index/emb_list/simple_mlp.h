// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#ifndef SIMPLE_MLP_H
#define SIMPLE_MLP_H

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#include "index/emb_list/cblas_decl.h"
#include "knowhere/log.h"

// OpenBLAS thread control
extern "C" {
void
openblas_set_num_threads(int num_threads);
int
openblas_get_num_threads(void);
}

namespace knowhere {

// RAII guard for openblas_set_num_threads: restores previous value on scope exit.
class ScopedBLASThreads {
    int old_;

 public:
    explicit ScopedBLASThreads(int n) : old_(openblas_get_num_threads()) {
        openblas_set_num_threads(n);
    }
    ~ScopedBLASThreads() {
        openblas_set_num_threads(old_);
    }
    ScopedBLASThreads(const ScopedBLASThreads&) = delete;
    ScopedBLASThreads&
    operator=(const ScopedBLASThreads&) = delete;
};

/**
 * @brief MLP implementation matching LEMUR paper/github.
 *
 * Architecture (num_layers=2, default):
 *   input -> [Linear -> LayerNorm -> GELU] -> [Linear -> LayerNorm -> GELU] -> Linear(no bias) -> output
 *            |<-------------- feature_extractor ---------------->|            |<-- output_layer -->|
 *
 * For LEMUR:
 *   - feature_extractor: extracts hidden features for query encoding
 *   - output_layer weights (W2): document representations [num_docs, final_hidden_dim]
 */
class SimpleMLP {
 public:
    /**
     * @brief Construct MLP matching LEMUR architecture.
     *
     * @param input_dim Input dimension (e.g., 128 for ColBERT)
     * @param output_dim Output dimension (num_docs)
     * @param hidden_dim Hidden layer dimension (default 1024)
     * @param final_hidden_dim Final hidden dimension before output (default same as hidden_dim)
     * @param num_layers Number of layers in feature_extractor (default 2)
     * @param seed Random seed for weight initialization
     */
    SimpleMLP(int32_t input_dim, int32_t output_dim, int32_t hidden_dim = 1024, int32_t final_hidden_dim = 0,
              int32_t num_layers = 2, int32_t seed = 42)
        : input_dim_(input_dim),
          output_dim_(output_dim),
          hidden_dim_(hidden_dim),
          num_layers_(num_layers),
          seed_(seed) {
        if (final_hidden_dim <= 0) {
            final_hidden_dim_ = hidden_dim;
        } else {
            final_hidden_dim_ = final_hidden_dim;
        }

        // Build layer_dims_ (network structure only, no weight allocation)
        for (int32_t layer = 0; layer < num_layers_; ++layer) {
            int32_t out_dim = (layer == num_layers_ - 1) ? final_hidden_dim_ : hidden_dim_;
            layer_dims_.push_back(out_dim);
        }
    }

    // Construct from deserialized weights (no training needed)
    SimpleMLP(int32_t input_dim, int32_t output_dim, int32_t hidden_dim, int32_t final_hidden_dim, int32_t num_layers,
              std::vector<std::vector<float>>&& fc_weights, std::vector<std::vector<float>>&& fc_biases,
              std::vector<std::vector<float>>&& ln_gammas, std::vector<std::vector<float>>&& ln_betas)
        : SimpleMLP(input_dim, output_dim, hidden_dim, final_hidden_dim, num_layers, 0) {
        fc_weights_ = std::move(fc_weights);
        fc_biases_ = std::move(fc_biases);
        ln_gammas_ = std::move(ln_gammas);
        ln_betas_ = std::move(ln_betas);
        trained_ = true;
    }

    /**
     * @brief Extract features (output of feature_extractor, before output_layer).
     *
     * @param input Input matrix [batch_size, input_dim]
     * @param batch_size Number of samples
     * @param features Output features [batch_size, final_hidden_dim]
     */
    void
    ExtractFeatures(const float* input, int32_t batch_size, float* features) {
        const float* current = input;
        int32_t current_dim = input_dim_;
        std::vector<float> linear_out, ln_out, act_out;

        for (int32_t layer = 0; layer < num_layers_; ++layer) {
            int32_t out_dim = layer_dims_[layer];
            linear_out.resize(batch_size * out_dim);
            ln_out.resize(batch_size * out_dim);
            act_out.resize(batch_size * out_dim);

            LinearForward(current, fc_weights_[layer].data(), fc_biases_[layer].data(), batch_size, current_dim,
                          out_dim, linear_out.data());

            LayerNormForward(linear_out.data(), ln_gammas_[layer].data(), ln_betas_[layer].data(), batch_size, out_dim,
                             ln_out.data());

            GELUForward(ln_out.data(), batch_size * out_dim, act_out.data());

            current = act_out.data();
            current_dim = out_dim;
        }

        std::memcpy(features, current, batch_size * final_hidden_dim_ * sizeof(float));
    }

 private:
    // Forward pass (training only). AllocateTrainingBuffers must be called first.
    // Backward reads directly from buf_linear_out_/buf_ln_out_/buf_act_out_.
    void
    Forward(const float* input, int32_t batch_size, float* output) {
        const float* current = input;
        int32_t current_dim = input_dim_;

        for (int32_t layer = 0; layer < num_layers_; ++layer) {
            int32_t out_dim = layer_dims_[layer];
            float* linear_out = buf_linear_out_[layer].data();
            float* ln_out = buf_ln_out_[layer].data();
            float* act_out = buf_act_out_[layer].data();

            LinearForward(current, fc_weights_[layer].data(), fc_biases_[layer].data(), batch_size, current_dim,
                          out_dim, linear_out);
            LayerNormForward(linear_out, ln_gammas_[layer].data(), ln_betas_[layer].data(), batch_size, out_dim,
                             ln_out);
            GELUForward(ln_out, batch_size * out_dim, act_out);

            current = act_out;
            current_dim = out_dim;
        }

        LinearForwardNoBias(current, W_out_.data(), batch_size, final_hidden_dim_, output_dim_, output);
    }

    /**
     * @brief Backward pass and compute gradients.
     *
     * @param input Input matrix [batch_size, input_dim]
     * @param target Target values [batch_size, output_dim]
     * @param output Predicted values [batch_size, output_dim]
     * @param batch_size Number of samples
     * @return MSE loss
     */
    float
    Backward(const float* input, const float* target, const float* output, int32_t batch_size) {
        // Gradients are zeroed inline: sgemm beta=0.0 overwrites dW, and db/dgamma/dbeta
        // are zeroed at the start of LinearBackward/LayerNormBackward.

        float total_loss = 0.0f;
        float scale = 1.0f / batch_size;

        // Compute output gradient: d_output = 2 * (output - target) / output_dim
        float* d_output = buf_d_output_.data();
        for (int32_t b = 0; b < batch_size; ++b) {
            for (int32_t j = 0; j < output_dim_; ++j) {
                float diff = output[b * output_dim_ + j] - target[b * output_dim_ + j];
                total_loss += diff * diff;
                d_output[b * output_dim_ + j] = 2.0f * diff * scale / output_dim_;
            }
        }

        // Backward through output layer (no bias)
        float* d_hidden = buf_d_hidden_.data();
        LinearBackwardNoBias(buf_act_out_[num_layers_ - 1].data(), d_output, W_out_.data(), batch_size,
                             final_hidden_dim_, output_dim_, dW_out_.data(), d_hidden);

        float* d_current = d_hidden;

        // Backward through feature_extractor layers (reverse order)
        for (int32_t layer = num_layers_ - 1; layer >= 0; --layer) {
            int32_t out_dim = layer_dims_[layer];
            int32_t in_dim = (layer == 0) ? input_dim_ : layer_dims_[layer - 1];

            const float* layer_input = (layer == 0) ? input : buf_act_out_[layer - 1].data();
            const float* pre_ln = buf_linear_out_[layer].data();
            const float* post_ln = buf_ln_out_[layer].data();

            float* d_act = buf_d_act_[layer].data();
            float* d_ln = buf_d_ln_[layer].data();
            float* d_input_layer = buf_d_input_[layer].data();

            GELUBackward(post_ln, d_current, batch_size * out_dim, d_act);

            LayerNormBackward(pre_ln, d_act, ln_gammas_[layer].data(), batch_size, out_dim, d_ln,
                              d_ln_gammas_[layer].data(), d_ln_betas_[layer].data());

            LinearBackward(layer_input, d_ln, fc_weights_[layer].data(), batch_size, in_dim, out_dim,
                           d_fc_weights_[layer].data(), d_fc_biases_[layer].data(), d_input_layer);

            d_current = d_input_layer;
        }

        return total_loss / (batch_size * output_dim_);
    }

    /**
     * @brief Update weights using Adam optimizer.
     */
    void
    UpdateAdam(float lr, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f) {
        adam_t_++;
        float bc1 = 1.0f - std::pow(beta1, adam_t_);
        float bc2 = 1.0f - std::pow(beta2, adam_t_);

        // Update feature_extractor layers
        for (int32_t layer = 0; layer < num_layers_; ++layer) {
            AdamUpdate(fc_weights_[layer], d_fc_weights_[layer], m_fc_weights_[layer], v_fc_weights_[layer], lr, beta1,
                       beta2, eps, bc1, bc2);
            AdamUpdate(fc_biases_[layer], d_fc_biases_[layer], m_fc_biases_[layer], v_fc_biases_[layer], lr, beta1,
                       beta2, eps, bc1, bc2);
            AdamUpdate(ln_gammas_[layer], d_ln_gammas_[layer], m_ln_gammas_[layer], v_ln_gammas_[layer], lr, beta1,
                       beta2, eps, bc1, bc2);
            AdamUpdate(ln_betas_[layer], d_ln_betas_[layer], m_ln_betas_[layer], v_ln_betas_[layer], lr, beta1, beta2,
                       eps, bc1, bc2);
        }

        // Update output layer
        AdamUpdate(W_out_, dW_out_, m_W_out_, v_W_out_, lr, beta1, beta2, eps, bc1, bc2);
    }

 public:
    /**
     * @brief Train on a dataset.
     */
    float
    Train(const float* X_train, const float* y_train, int32_t num_samples, int32_t epochs = 100,
          int32_t batch_size = 512, float lr = 0.001f) {
        // 4 is based on some experiments
        int blas_threads = std::min(static_cast<int>(std::thread::hardware_concurrency()), 4);
        if (blas_threads < 1)
            blas_threads = 1;
        ScopedBLASThreads blas_guard(blas_threads);

        // Guard: do not retrain a deserialized model
        if (trained_) {
            LOG_KNOWHERE_WARNING_ << "[LEMUR MLP] Model already trained/loaded, skipping Train()";
            return 0.0f;
        }

        // Initialize weights (first training call)
        InitWeights();

        // Rebuild gradient/Adam states if previously released
        if (dW_out_.empty()) {
            InitGradients();
            InitAdamStates();
            adam_t_ = 0;
        }

        // Pre-allocate all training buffers to avoid malloc in hot loop
        AllocateTrainingBuffers(batch_size);

        std::vector<int64_t> indices(num_samples);
        for (int64_t i = 0; i < num_samples; ++i) {
            indices[i] = i;
        }

        std::vector<float> batch_input(batch_size * input_dim_);
        std::vector<float> batch_target(batch_size * output_dim_);
        std::vector<float> batch_output(batch_size * output_dim_);

        std::mt19937 rng(seed_);
        float final_loss = 0.0f;

        LOG_KNOWHERE_INFO_ << "[LEMUR MLP] Training started: samples=" << num_samples << ", epochs=" << epochs
                           << ", batch_size=" << batch_size << ", lr=" << lr;

        // Early stopping parameters
        float best_loss = std::numeric_limits<float>::max();
        int32_t patience_counter = 0;
        const int32_t patience = 5;     // Stop if no improvement for 5 epochs
        const float min_delta = 1e-4f;  // Minimum improvement to reset patience
        const int32_t log_interval = 3;

        for (int32_t epoch = 0; epoch < epochs; ++epoch) {
            auto epoch_start = std::chrono::high_resolution_clock::now();
            std::shuffle(indices.begin(), indices.end(), rng);

            float epoch_loss = 0.0f;
            int32_t num_batches = 0;

            for (int32_t start = 0; start < num_samples; start += batch_size) {
                int32_t actual_batch = std::min(batch_size, num_samples - start);

                // Gather batch
                for (int32_t b = 0; b < actual_batch; ++b) {
                    int32_t idx = indices[start + b];
                    std::memcpy(batch_input.data() + b * input_dim_, X_train + idx * input_dim_,
                                input_dim_ * sizeof(float));
                    std::memcpy(batch_target.data() + b * output_dim_, y_train + idx * output_dim_,
                                output_dim_ * sizeof(float));
                }

                Forward(batch_input.data(), actual_batch, batch_output.data());

                // Backward
                float loss = Backward(batch_input.data(), batch_target.data(), batch_output.data(), actual_batch);
                epoch_loss += loss;
                num_batches++;

                // Update
                UpdateAdam(lr);
            }

            final_loss = epoch_loss / num_batches;

            auto epoch_end = std::chrono::high_resolution_clock::now();
            double epoch_ms = std::chrono::duration<double, std::milli>(epoch_end - epoch_start).count();

            // Log training progress
            bool should_log = (epoch == 0) || (epoch == epochs - 1) || ((epoch + 1) % log_interval == 0);
            if (should_log) {
                LOG_KNOWHERE_DEBUG_ << "[LEMUR MLP] Epoch " << (epoch + 1) << "/" << epochs << ", Loss: " << final_loss
                                    << ", Time: " << epoch_ms << " ms";
            }

            // Early stopping check
            if (best_loss - final_loss > min_delta) {
                best_loss = final_loss;
                patience_counter = 0;
            } else {
                patience_counter++;
                if (patience_counter >= patience) {
                    LOG_KNOWHERE_INFO_ << "[LEMUR MLP] Early stopping at epoch " << (epoch + 1)
                                       << ", loss=" << final_loss;
                    break;
                }
            }
        }

        LOG_KNOWHERE_INFO_ << "[LEMUR MLP] Training completed, final loss: " << final_loss;

        trained_ = true;
        return final_loss;
    }

    int32_t
    FinalHiddenDim() const {
        return final_hidden_dim_;
    }
    // For serialization
    const std::vector<std::vector<float>>&
    GetFcWeights() const {
        return fc_weights_;
    }
    const std::vector<std::vector<float>>&
    GetFcBiases() const {
        return fc_biases_;
    }
    const std::vector<std::vector<float>>&
    GetLnGammas() const {
        return ln_gammas_;
    }
    const std::vector<std::vector<float>>&
    GetLnBetas() const {
        return ln_betas_;
    }

    /**
     * @brief Release memory used only during training.
     *
     * Frees gradients, Adam optimizer states, intermediate activations,
     * and pre-allocated training buffers. Call after training is complete.
     */
    void
    ReleaseTrainingBuffers() {
        // Gradients
        d_fc_weights_.clear();
        d_fc_weights_.shrink_to_fit();
        d_fc_biases_.clear();
        d_fc_biases_.shrink_to_fit();
        d_ln_gammas_.clear();
        d_ln_gammas_.shrink_to_fit();
        d_ln_betas_.clear();
        d_ln_betas_.shrink_to_fit();
        dW_out_.clear();
        dW_out_.shrink_to_fit();

        // Adam states
        m_fc_weights_.clear();
        m_fc_weights_.shrink_to_fit();
        v_fc_weights_.clear();
        v_fc_weights_.shrink_to_fit();
        m_fc_biases_.clear();
        m_fc_biases_.shrink_to_fit();
        v_fc_biases_.clear();
        v_fc_biases_.shrink_to_fit();
        m_ln_gammas_.clear();
        m_ln_gammas_.shrink_to_fit();
        v_ln_gammas_.clear();
        v_ln_gammas_.shrink_to_fit();
        m_ln_betas_.clear();
        m_ln_betas_.shrink_to_fit();
        v_ln_betas_.clear();
        v_ln_betas_.shrink_to_fit();
        m_W_out_.clear();
        m_W_out_.shrink_to_fit();
        v_W_out_.clear();
        v_W_out_.shrink_to_fit();

        // Training buffers
        buf_linear_out_.clear();
        buf_linear_out_.shrink_to_fit();
        buf_ln_out_.clear();
        buf_ln_out_.shrink_to_fit();
        buf_act_out_.clear();
        buf_act_out_.shrink_to_fit();
        buf_d_output_.clear();
        buf_d_output_.shrink_to_fit();
        buf_d_hidden_.clear();
        buf_d_hidden_.shrink_to_fit();
        buf_d_act_.clear();
        buf_d_act_.shrink_to_fit();
        buf_d_ln_.clear();
        buf_d_ln_.shrink_to_fit();
        buf_d_input_.clear();
        buf_d_input_.shrink_to_fit();
        buf_x_norm_.clear();
        buf_x_norm_.shrink_to_fit();

        allocated_batch_size_ = 0;
    }

 private:
    int32_t input_dim_;
    int32_t output_dim_;
    int32_t hidden_dim_;
    int32_t final_hidden_dim_;
    int32_t num_layers_;
    int32_t seed_;
    bool trained_ = false;

    // Feature extractor layers
    std::vector<std::vector<float>> fc_weights_;  // [num_layers][out_dim * in_dim]
    std::vector<std::vector<float>> fc_biases_;   // [num_layers][out_dim]
    std::vector<std::vector<float>> ln_gammas_;   // [num_layers][out_dim]
    std::vector<std::vector<float>> ln_betas_;    // [num_layers][out_dim]
    std::vector<int32_t> layer_dims_;             // output dim of each layer

    // Output layer (no bias, matching LEMUR)
    std::vector<float> W_out_;  // [output_dim * final_hidden_dim]

    // Gradients
    std::vector<std::vector<float>> d_fc_weights_;
    std::vector<std::vector<float>> d_fc_biases_;
    std::vector<std::vector<float>> d_ln_gammas_;
    std::vector<std::vector<float>> d_ln_betas_;
    std::vector<float> dW_out_;

    // Adam states
    std::vector<std::vector<float>> m_fc_weights_, v_fc_weights_;
    std::vector<std::vector<float>> m_fc_biases_, v_fc_biases_;
    std::vector<std::vector<float>> m_ln_gammas_, v_ln_gammas_;
    std::vector<std::vector<float>> m_ln_betas_, v_ln_betas_;
    std::vector<float> m_W_out_, v_W_out_;
    int32_t adam_t_ = 0;

    // ========== Pre-allocated buffers for training (avoid malloc in hot loop) ==========
    int32_t allocated_batch_size_ = 0;  // Current allocated batch size

    // Forward pass buffers (per layer)
    std::vector<std::vector<float>> buf_linear_out_;  // [num_layers][batch * layer_dim]
    std::vector<std::vector<float>> buf_ln_out_;      // [num_layers][batch * layer_dim]
    std::vector<std::vector<float>> buf_act_out_;     // [num_layers][batch * layer_dim]
    // Backward pass buffers
    std::vector<float> buf_d_output_;              // [batch * output_dim]
    std::vector<float> buf_d_hidden_;              // [batch * final_hidden_dim]
    std::vector<std::vector<float>> buf_d_act_;    // [num_layers][batch * layer_dim]
    std::vector<std::vector<float>> buf_d_ln_;     // [num_layers][batch * layer_dim]
    std::vector<std::vector<float>> buf_d_input_;  // [num_layers][batch * in_dim]

    // LayerNorm backward buffer
    std::vector<float> buf_x_norm_;  // [max_dim]

    /**
     * @brief Randomly initialize all weights (feature_extractor + output layer).
     *
     * Called lazily at the start of Train(). Not called during deserialization
     * since weights are loaded via Set* methods.
     */
    void
    InitWeights() {
        std::mt19937 rng(seed_);

        // Input dims per layer: [input_dim, hidden_dim, hidden_dim, ...]
        std::vector<int32_t> dims;
        dims.push_back(input_dim_);
        for (int32_t i = 0; i < num_layers_ - 1; ++i) {
            dims.push_back(hidden_dim_);
        }

        fc_weights_.resize(num_layers_);
        fc_biases_.resize(num_layers_);
        ln_gammas_.resize(num_layers_);
        ln_betas_.resize(num_layers_);

        for (int32_t layer = 0; layer < num_layers_; ++layer) {
            int32_t in_dim = dims[layer];
            int32_t out_dim = layer_dims_[layer];

            float scale = std::sqrt(2.0f / (in_dim + out_dim));
            std::normal_distribution<float> dist(0.0f, scale);

            fc_weights_[layer].resize(out_dim * in_dim);
            for (auto& w : fc_weights_[layer]) {
                w = dist(rng);
            }
            fc_biases_[layer].assign(out_dim, 0.0f);
            ln_gammas_[layer].assign(out_dim, 1.0f);
            ln_betas_[layer].assign(out_dim, 0.0f);
        }

        // Output layer: [output_dim, final_hidden_dim], no bias
        float scale_out = std::sqrt(2.0f / (final_hidden_dim_ + output_dim_));
        std::normal_distribution<float> dist_out(0.0f, scale_out);
        W_out_.resize(output_dim_ * final_hidden_dim_);
        for (auto& w : W_out_) {
            w = dist_out(rng);
        }
    }

    void
    InitGradients() {
        d_fc_weights_.resize(num_layers_);
        d_fc_biases_.resize(num_layers_);
        d_ln_gammas_.resize(num_layers_);
        d_ln_betas_.resize(num_layers_);

        for (int32_t i = 0; i < num_layers_; ++i) {
            d_fc_weights_[i].resize(fc_weights_[i].size(), 0.0f);
            d_fc_biases_[i].resize(fc_biases_[i].size(), 0.0f);
            d_ln_gammas_[i].resize(ln_gammas_[i].size(), 0.0f);
            d_ln_betas_[i].resize(ln_betas_[i].size(), 0.0f);
        }

        dW_out_.resize(W_out_.size(), 0.0f);
    }

    void
    InitAdamStates() {
        m_fc_weights_.resize(num_layers_);
        v_fc_weights_.resize(num_layers_);
        m_fc_biases_.resize(num_layers_);
        v_fc_biases_.resize(num_layers_);
        m_ln_gammas_.resize(num_layers_);
        v_ln_gammas_.resize(num_layers_);
        m_ln_betas_.resize(num_layers_);
        v_ln_betas_.resize(num_layers_);

        for (int32_t i = 0; i < num_layers_; ++i) {
            m_fc_weights_[i].resize(fc_weights_[i].size(), 0.0f);
            v_fc_weights_[i].resize(fc_weights_[i].size(), 0.0f);
            m_fc_biases_[i].resize(fc_biases_[i].size(), 0.0f);
            v_fc_biases_[i].resize(fc_biases_[i].size(), 0.0f);
            m_ln_gammas_[i].resize(ln_gammas_[i].size(), 0.0f);
            v_ln_gammas_[i].resize(ln_gammas_[i].size(), 0.0f);
            m_ln_betas_[i].resize(ln_betas_[i].size(), 0.0f);
            v_ln_betas_[i].resize(ln_betas_[i].size(), 0.0f);
        }

        m_W_out_.resize(W_out_.size(), 0.0f);
        v_W_out_.resize(W_out_.size(), 0.0f);
    }

    /**
     * @brief Pre-allocate buffers for training to avoid malloc in hot loop.
     */
    void
    AllocateTrainingBuffers(int32_t batch_size) {
        if (batch_size <= allocated_batch_size_) {
            return;  // Already allocated enough
        }

        allocated_batch_size_ = batch_size;

        // Find max dimension across all layers
        int32_t max_dim = input_dim_;
        for (int32_t i = 0; i < num_layers_; ++i) {
            max_dim = std::max(max_dim, layer_dims_[i]);
        }

        // Forward pass buffers
        buf_linear_out_.resize(num_layers_);
        buf_ln_out_.resize(num_layers_);
        buf_act_out_.resize(num_layers_);
        for (int32_t i = 0; i < num_layers_; ++i) {
            buf_linear_out_[i].resize(batch_size * layer_dims_[i]);
            buf_ln_out_[i].resize(batch_size * layer_dims_[i]);
            buf_act_out_[i].resize(batch_size * layer_dims_[i]);
        }
        // Backward pass buffers
        buf_d_output_.resize(batch_size * output_dim_);
        buf_d_hidden_.resize(batch_size * final_hidden_dim_);
        buf_d_act_.resize(num_layers_);
        buf_d_ln_.resize(num_layers_);
        buf_d_input_.resize(num_layers_);
        for (int32_t i = 0; i < num_layers_; ++i) {
            int32_t in_dim = (i == 0) ? input_dim_ : layer_dims_[i - 1];
            buf_d_act_[i].resize(batch_size * layer_dims_[i]);
            buf_d_ln_[i].resize(batch_size * layer_dims_[i]);
            buf_d_input_[i].resize(batch_size * in_dim);
        }

        // LayerNorm buffer
        buf_x_norm_.resize(max_dim);
    }

    // ========== Layer Operations (BLAS optimized) ==========

    // Linear: Y = X @ W.T + b
    // X: [batch, in_dim], W: [out_dim, in_dim], Y: [batch, out_dim]
    void
    LinearForward(const float* X, const float* W, const float* b, int32_t batch, int32_t in_dim, int32_t out_dim,
                  float* Y) {
        // Y = X @ W.T using cblas_sgemm
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch, out_dim, in_dim, 1.0f, X, in_dim, W, in_dim, 0.0f,
                    Y, out_dim);

        // Add bias: Y[i,:] += b
        for (int32_t i = 0; i < batch; ++i) {
            cblas_saxpy(out_dim, 1.0f, b, 1, Y + i * out_dim, 1);
        }
    }

    // Linear without bias: Y = X @ W.T
    void
    LinearForwardNoBias(const float* X, const float* W, int32_t batch, int32_t in_dim, int32_t out_dim, float* Y) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch, out_dim, in_dim, 1.0f, X, in_dim, W, in_dim, 0.0f,
                    Y, out_dim);
    }

    // Linear backward
    // dW: [out_dim, in_dim], dY: [batch, out_dim], X: [batch, in_dim], W: [out_dim, in_dim]
    void
    LinearBackward(const float* X, const float* dY, const float* W, int32_t batch, int32_t in_dim, int32_t out_dim,
                   float* dW, float* db, float* dX) {
        // dW = dY.T @ X (beta=0.0 overwrites, no need for separate ClearGradients)
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, out_dim, in_dim, batch, 1.0f, dY, out_dim, X, in_dim, 0.0f,
                    dW, in_dim);

        // db = sum(dY, axis=0)
        std::fill(db, db + out_dim, 0.0f);
        for (int32_t i = 0; i < batch; ++i) {
            cblas_saxpy(out_dim, 1.0f, dY + i * out_dim, 1, db, 1);
        }

        // dX = dY @ W
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, batch, in_dim, out_dim, 1.0f, dY, out_dim, W, in_dim,
                    0.0f, dX, in_dim);
    }

    // Linear backward without bias
    void
    LinearBackwardNoBias(const float* X, const float* dY, const float* W, int32_t batch, int32_t in_dim,
                         int32_t out_dim, float* dW, float* dX) {
        // dW = dY.T @ X (beta=0.0 overwrites, no need for separate ClearGradients)
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, out_dim, in_dim, batch, 1.0f, dY, out_dim, X, in_dim, 0.0f,
                    dW, in_dim);

        // dX = dY @ W
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, batch, in_dim, out_dim, 1.0f, dY, out_dim, W, in_dim,
                    0.0f, dX, in_dim);
    }

    // LayerNorm forward: Y = (X - mean) / std * gamma + beta
    void
    LayerNormForward(const float* X, const float* gamma, const float* beta, int32_t batch, int32_t dim, float* Y) {
        const float eps = 1e-5f;

        for (int32_t i = 0; i < batch; ++i) {
            const float* x_row = X + i * dim;
            float* y_row = Y + i * dim;

            float mean = 0.0f;
            for (int32_t j = 0; j < dim; ++j) {
                mean += x_row[j];
            }
            mean /= dim;

            float var = 0.0f;
            for (int32_t j = 0; j < dim; ++j) {
                float diff = x_row[j] - mean;
                var += diff * diff;
            }
            var /= dim;

            float inv_std = 1.0f / std::sqrt(var + eps);

            for (int32_t j = 0; j < dim; ++j) {
                y_row[j] = (x_row[j] - mean) * inv_std * gamma[j] + beta[j];
            }
        }
    }

    // LayerNorm backward
    void
    LayerNormBackward(const float* X, const float* dY, const float* gamma, int32_t batch, int32_t dim, float* dX,
                      float* dgamma, float* dbeta) {
        const float eps = 1e-5f;

        // Use pre-allocated buffer if available, otherwise use member buffer
        float* x_norm;
        if (static_cast<int32_t>(buf_x_norm_.size()) >= dim) {
            x_norm = buf_x_norm_.data();
        } else {
            buf_x_norm_.resize(dim);
            x_norm = buf_x_norm_.data();
        }

        // Zero dgamma/dbeta before accumulation (replaces ClearGradients)
        std::fill(dgamma, dgamma + dim, 0.0f);
        std::fill(dbeta, dbeta + dim, 0.0f);

        for (int32_t i = 0; i < batch; ++i) {
            const float* x_row = X + i * dim;
            const float* dy_row = dY + i * dim;
            float* dx_row = dX + i * dim;

            float mean = 0.0f;
            for (int32_t j = 0; j < dim; ++j) {
                mean += x_row[j];
            }
            mean /= dim;

            float var = 0.0f;
            for (int32_t j = 0; j < dim; ++j) {
                float diff = x_row[j] - mean;
                var += diff * diff;
            }
            var /= dim;

            float inv_std = 1.0f / std::sqrt(var + eps);

            for (int32_t j = 0; j < dim; ++j) {
                x_norm[j] = (x_row[j] - mean) * inv_std;
            }

            for (int32_t j = 0; j < dim; ++j) {
                dgamma[j] += dy_row[j] * x_norm[j];
            }
            cblas_saxpy(dim, 1.0f, dy_row, 1, dbeta, 1);

            float sum_dy_gamma = 0.0f;
            float sum_dy_gamma_xnorm = 0.0f;
            for (int32_t j = 0; j < dim; ++j) {
                float dy_g = dy_row[j] * gamma[j];
                sum_dy_gamma += dy_g;
                sum_dy_gamma_xnorm += dy_g * x_norm[j];
            }

            float scale = inv_std / dim;
            for (int32_t j = 0; j < dim; ++j) {
                dx_row[j] = scale * (dim * dy_row[j] * gamma[j] - sum_dy_gamma - x_norm[j] * sum_dy_gamma_xnorm);
            }
        }
    }

    // Fast tanh approximation using Pade approximation
    // Accurate to ~1e-4 in range [-3, 3], saturates outside
    static inline float
    fast_tanh(float x) {
        if (x < -3.0f)
            return -1.0f;
        if (x > 3.0f)
            return 1.0f;
        float x2 = x * x;
        return x * (27.0f + x2) / (27.0f + 9.0f * x2);
    }

    // GELU forward: GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    void
    GELUForward(const float* X, int32_t n, float* Y) {
        const float sqrt_2_over_pi = 0.7978845608028654f;
        const float coeff = 0.044715f;

        for (int32_t i = 0; i < n; ++i) {
            float x = X[i];
            float x3 = x * x * x;
            float inner = sqrt_2_over_pi * (x + coeff * x3);
            Y[i] = 0.5f * x * (1.0f + fast_tanh(inner));
        }
    }

    // GELU backward
    void
    GELUBackward(const float* X, const float* dY, int32_t n, float* dX) {
        const float sqrt_2_over_pi = 0.7978845608028654f;
        const float coeff = 0.044715f;
        const float coeff3 = 3.0f * coeff;

        for (int32_t i = 0; i < n; ++i) {
            float x = X[i];
            float x2 = x * x;
            float x3 = x2 * x;
            float inner = sqrt_2_over_pi * (x + coeff * x3);
            float tanh_inner = fast_tanh(inner);
            float sech2 = 1.0f - tanh_inner * tanh_inner;
            float d_inner = sqrt_2_over_pi * (1.0f + coeff3 * x2);
            dX[i] = dY[i] * (0.5f * (1.0f + tanh_inner) + 0.5f * x * sech2 * d_inner);
        }
    }

    // Adam update
    void
    AdamUpdate(std::vector<float>& param, std::vector<float>& grad, std::vector<float>& m, std::vector<float>& v,
               float lr, float beta1, float beta2, float eps, float bc1, float bc2) {
        const size_t n = param.size();
        const float one_minus_beta1 = 1.0f - beta1;
        const float one_minus_beta2 = 1.0f - beta2;
        const float lr_bc1 = lr / bc1;
        const float inv_bc2 = 1.0f / bc2;

        for (size_t i = 0; i < n; ++i) {
            m[i] = beta1 * m[i] + one_minus_beta1 * grad[i];
            v[i] = beta2 * v[i] + one_minus_beta2 * grad[i] * grad[i];
            float v_hat = v[i] * inv_bc2;
            param[i] -= lr_bc1 * m[i] / (std::sqrt(v_hat) + eps);
        }
    }
};

}  // namespace knowhere

#endif /* SIMPLE_MLP_H */
