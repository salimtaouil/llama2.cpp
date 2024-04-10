#include "nn_blocks.h"
#include "transformer.h"
#include <vector>
#include <cmath>
#include <iostream>

void copy(tensor1d &dst, tensor1d &src) {
    for (int i = 0; i < dst.size(); i++)  dst[i] = src[i];
}
void copy(tensor2d &dst, tensor2d &src) {
    for (int i = 0; i < dst.size(); i++)  copy(dst[i], src[i]);
}
void copy(tensor3d &dst, tensor3d &src) {
    for (int i = 0; i < dst.size(); i++)  copy(dst[i], src[i]);
}

void accum(tensor1d &lhs, tensor1d &rhs) {
    for (int i = 0; i < rhs.size(); ++i)  lhs[i] += rhs[i];
}

void rmsnorm(tensor1d &output, tensor1d &input, tensor1d &weight) {
    float ss = 0.0;
    for (int i = 0; i < input.size(); i++)
        ss += input[i] * input[i];
    ss = ss / input.size() + EPS;
    float inv_ss = 1 / sqrt(ss);
    for (int i = 0; i < input.size(); i++)
        output[i] = input[i] * inv_ss * weight[i];
}

void softmax(tensor1d &output, tensor1d &input, int max_pos = -1) {
    if (max_pos == -1)  max_pos = input.size();
    float max_val = input[0];
    for (int i = 1; i < max_pos; i++)
        if (input[i] > max_val)  max_val = input[i];
    
    // exp and sum
    float sum = 0;
    for (int i = 0; i < max_pos; i++) {
        output[i] = exp(input[i] - max_val);
        sum += output[i];
    }
    // normalize
    for (int i = 0; i < max_pos; i++)
        output[i] /= sum;
}

void matmul(tensor1d &output, tensor1d &input, tensor2d &weight) {
    for (int i = 0; i < output.size(); i++) {
        output[i] = 0;
        for (int j = 0; j < input.size(); j++)
            output[i] += input[j] * weight[i][j];
    }
}

void transformer(int token_index, int token_position, Config &config, RunState &state, TransformerWeights &transformer_weights) {
    // a few convenience variables
    int dim = config.dim;
    int hidden_dim = config.hidden_dim;
    int head_size = dim / config.n_heads;

    // copy the token embedding into x
    copy(state.x, transformer_weights.token_embedding_table[token_index]);

    for (int layer = 0; layer < config.n_layers; ++layer) {
        // attention rmsnorm
        rmsnorm(state.xb, state.x, transformer_weights.rms_att_weight[layer]);

        // attention
        matmul(state.q, state.xb, transformer_weights.wq[layer]);
        matmul(state.k, state.xb, transformer_weights.wk[layer]);
        matmul(state.v, state.xb, transformer_weights.wv[layer]);

        // apply RoPE positional embeddings
        for (int head = 0; head < config.n_heads; ++head) {
            int start = head * head_size;
            for (int i = 0; i < head_size; i += 2) {
                float q0 = state.q[start + i];
                float q1 = state.q[start + i + 1];
                float k0 = state.k[start + i];
                float k1 = state.k[start + i + 1];
                float fcr = transformer_weights.freq_cis_real[token_position][i / 2];
                float fci = transformer_weights.freq_cis_imag[token_position][i / 2];
                state.q[start + i]     = q0 * fcr - q1 * fci;
                state.q[start + i + 1] = q0 * fci + q1 * fcr;
                state.k[start + i]     = k0 * fcr - k1 * fci;
                state.k[start + i + 1] = k0 * fci + k1 * fcr;
            }
        }

        // save key/value in cache
        copy(state.key_cache[layer][token_position], state.k);
        copy(state.value_cache[layer][token_position], state.v);

        // multiquery attention
        for (int head = 0; head < config.n_heads; ++head) {
            for (int timestep = 0; timestep < token_position; ++timestep) {
                float score = 0;
                for (int i = 0; i < head_size; ++i)
                    score += state.q[head * head_size + i] * state.key_cache[layer][timestep][head * head_size + i];
                score /= std::sqrt(head_size * 1.0);
                state.attention[timestep] = score;
            }

            // softmax
            softmax(state.attention, state.attention, token_position+1);

            // weighted sum
            for (int i = 0; i < head_size; ++i) {
                state.xb[head * head_size + i] = 0;
                for (int timestep = 0; timestep <= token_position; ++timestep)
                    state.xb[head * head_size + i] += state.attention[timestep] * state.value_cache[layer][timestep][head * head_size + i];
            }
        }

        // final matmul to get the output of the attention
        matmul(state.xb2, state.xb, transformer_weights.wo[layer]);

        // residual connection back into x
        accum(state.x, state.xb2);

        // ffn rmsnorm
        rmsnorm(state.xb, state.x, transformer_weights.rms_ffn_weight[layer]);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x))) * self.w3(x)
        // first calculate self.w1(x) and self.w3(x)
        matmul(state.hb, state.xb, transformer_weights.w1[layer]);
        matmul(state.hb2, state.xb, transformer_weights.w3[layer]);

        // F.silu; silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
        for (int i = 0; i < hidden_dim; ++i)
            state.hb[i] = state.hb[i] * (1.0 / (1.0 + std::exp(-state.hb[i])));

        // elementwise multiple with w3(x)
        for (int i = 0; i < hidden_dim; ++i)
            state.hb[i] = state.hb[i] * state.hb2[i];
        
        // final matmul to get the output of the ffn
        matmul(state.xb, state.hb, transformer_weights.w2[layer]);

        // residual connection
        accum(state.x, state.xb);
    }

    // final rmsnorm
    rmsnorm(state.x, state.x, transformer_weights.rms_final_weight);

    // classifier into logits
    matmul(state.logits, state.x, transformer_weights.token_embedding_table);
}

