#include "transformer.h"
#include <vector>
#include <fstream>
#include <iostream>
#include "transformer.h"
#include <vector>
#include <fstream>
#include <iostream>

Transformer::Transformer(std::string checkpoint) : checkpoint_{checkpoint}
{
    ReadCheckPoint(checkpoint_);
    resize_state_tensors();
}

Transformer::~Transformer()
{

}

std::string Transformer::GetCheckPoint(){
    return checkpoint_;
}

void Transformer::resize_state_tensors() {
    tensor1d(config.dim).swap(state.x);
    tensor1d(config.dim).swap(state.xb);
    tensor1d(config.dim).swap(state.xb2);
    tensor1d(config.hidden_dim).swap(state.hb);
    tensor1d(config.hidden_dim).swap(state.hb2);
    tensor1d(config.dim).swap(state.q);
    tensor1d(config.dim).swap(state.k);
    tensor1d(config.dim).swap(state.v);
    tensor1d(config.seq_len).swap(state.attention);
    tensor1d(config.vocab_size).swap(state.logits);
    tensor3d(config.n_layers, tensor2d(config.seq_len, tensor1d(config.dim))).swap(state.key_cache);
    tensor3d(config.n_layers, tensor2d(config.seq_len, tensor1d(config.dim))).swap(state.value_cache);
}

void Transformer::resize_weights_tensors() {
    tensor2d(config.vocab_size, tensor1d(config.dim)).swap(weights.token_embedding_table);
    tensor2d(config.n_layers, tensor1d(config.dim)).swap(weights.rms_att_weight);
    tensor2d(config.n_layers, tensor1d(config.dim)).swap(weights.rms_ffn_weight);
    tensor3d(config.n_layers, tensor2d(config.dim, tensor1d(config.dim))).swap(weights.wq);
    tensor3d(config.n_layers, tensor2d(config.dim, tensor1d(config.dim))).swap(weights.wk);
    tensor3d(config.n_layers, tensor2d(config.dim, tensor1d(config.dim))).swap(weights.wv);
    tensor3d(config.n_layers, tensor2d(config.dim, tensor1d(config.dim))).swap(weights.wo);
    tensor3d(config.n_layers, tensor2d(config.hidden_dim, tensor1d(config.dim))).swap(weights.w1);
    tensor3d(config.n_layers, tensor2d(config.dim, tensor1d(config.hidden_dim))).swap(weights.w2);
    tensor3d(config.n_layers, tensor2d(config.hidden_dim, tensor1d(config.dim))).swap(weights.w3);
    tensor1d(config.dim).swap(weights.rms_final_weight);
    int head_size = config.dim / config.n_heads;
    tensor2d(config.seq_len, tensor1d(head_size / 2)).swap(weights.freq_cis_real);
    tensor2d(config.seq_len, tensor1d(head_size / 2)).swap(weights.freq_cis_imag);
}

void checkpoint_init_tensor(tensor1d &tensor, std::fstream &file) {
    file.read((char*)(tensor.data()), tensor.size() * sizeof(float));
}
void checkpoint_init_tensor(tensor2d &tensor, std::fstream &file) {
    for (auto &t : tensor) checkpoint_init_tensor(t, file);
}
void checkpoint_init_tensor(tensor3d &tensor, std::fstream &file) {
    for (auto &t : tensor) checkpoint_init_tensor(t, file);
}

void checkpoint_init_weights(TransformerWeights &weights, Config &config, std::fstream &file) {
    checkpoint_init_tensor(weights.token_embedding_table, file);
    checkpoint_init_tensor(weights.rms_att_weight, file);
    checkpoint_init_tensor(weights.wq, file);
    checkpoint_init_tensor(weights.wk, file);
    checkpoint_init_tensor(weights.wv, file);
    checkpoint_init_tensor(weights.wo, file);
    checkpoint_init_tensor(weights.rms_ffn_weight, file);
    checkpoint_init_tensor(weights.w1, file);
    checkpoint_init_tensor(weights.w2, file);
    checkpoint_init_tensor(weights.w3, file);
    checkpoint_init_tensor(weights.rms_final_weight, file);
    checkpoint_init_tensor(weights.freq_cis_real, file);
    checkpoint_init_tensor(weights.freq_cis_imag, file);
}

void Transformer::ReadCheckPoint(const std::string checkpoint) {
    std::fstream file(checkpoint);
    if (!file) {
        std::cout << "Unable to open the checkpoint file " << checkpoint << "\n";
        return;
    }
    // read file contents to config
    file.read((char*)&config, sizeof(config));
    resize_weights_tensors();
    checkpoint_init_weights(weights, config, file);
    file.close();
}