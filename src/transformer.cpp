#include "transformer.h"
#include <vector>
#include <fstream>
#include <iostream>

Transformer::Transformer(const char* checkpoint)
Transformer::Transformer(const char* checkpoint)
{
    ReadCheckPoint(checkpoint);

    resize_run_state();
    ReadCheckPoint(checkpoint);
    resize_state_tensors();
}

Transformer::~Transformer()
{
    if(this->data != MAP_FAILED){munmap(this->data, this->file_size);}
    if(*this->fd != -1) {close(*this->fd);}
}

void Transformer::resize_run_state()
{
    int kv_dim = (this->config.dim * this->config.n_kv_heads) / this->config.n_heads;
    this->state.x.resize(this->config.dim);
    this->state.xb.resize(this->config.dim);
    this->state.xb2.resize(this->config.dim);
    this->state.hb.resize(this->config.hidden_dim);
    this->state.hb2.resize(this->config.hidden_dim);
    this->state.q.resize(this->config.dim);
    this->state.key_cache.resize(this->config.n_layers * this->config.seq_len * kv_dim);
    this->state.value_cache.resize(this->config.n_layers * this->config.seq_len * kv_dim);
    this->state.att.resize(this->config.n_heads * this->config.seq_len);
    this->state.logits.resize(this->config.vocab_size);
}


void Transformer::ReadCheckPoint(const char* checkpoint)
{
    std::FILE* f = std::fopen(checkpoint, "r");

    if(!f)
        std::cout << "failed to open" << checkpoint << '\n';
    else
    {
        std::fread(&this->config, sizeof(this->config), 1, f);
        // negative vocab size is hacky way of signaling unshared weights. bit yikes.
        int shared_weights = this->config.vocab_size > 0 ? 1 : 0;
        this->config.vocab_size = std::abs(this->config.vocab_size);
        // figure out the file size
        std::fseek(f, 0, SEEK_END); // move file pointer to end of file
        this->file_size = std::ftell(f); // get the file size, in bytes
        std::fclose(f);
        // memory map the Transformer weights into the data pointer
        this->fd = std::make_unique<int>();
        *this->fd = open(checkpoint, O_RDONLY);
        if(*this->fd == -1)
            std::cout << "open failed!\n";
        else
        {
            this->data = (float*)mmap(NULL, this->file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
            if(this->data == MAP_FAILED) 
                std::cout << "mmap failed!\n";
            float* weights_ptr = this->data + sizeof(Config)/sizeof(float);
            memory_map_weights(weights_ptr,shared_weights);
        }
    }
}

void Transformer::memory_map_weights(float* ptr, int shared_weights){

    int head_size = this->config.dim / this->config.n_heads;

    unsigned long long n_layers = this->config.n_layers;
    std::vector<float> token_embedding_table{ptr, ptr + (this->config.vocab_size * this->config.dim)};
    this->weights.token_embedding_table = token_embedding_table;
    ptr += this->config.vocab_size * this->config.dim;
    std::vector<float> rms_att_weight{ptr, ptr + (n_layers * this->config.dim)};
    this->weights.rms_att_weight = rms_att_weight;
    ptr += n_layers * this->config.dim;
    std::vector<float> wq{ptr, ptr + (n_layers * this->config.dim * (this->config.n_heads * head_size))};
    this->weights.wq = wq;
    ptr += n_layers * this->config.dim * (this->config.n_heads * head_size);
    std::vector<float> wk{ptr, ptr + (n_layers * this->config.dim *(this->config.n_kv_heads * head_size))};
    this->weights.wk = wk;
    ptr += n_layers * this->config.dim *(this->config.n_kv_heads * head_size);
    std::vector<float> wv{ptr, ptr + (n_layers * this->config.dim * (this->config.n_kv_heads * head_size))};
    this->weights.wv = wv;
    ptr += n_layers * this->config.dim * (this->config.n_kv_heads * head_size);
    std::vector<float> wo{ptr, ptr + (n_layers * (this->config.n_heads * head_size) * this->config.dim)};
    this->weights.wo = wo;
    ptr += n_layers * (this->config.n_heads * head_size) * this->config.dim;
    std::vector<float> rms_ffn_weight{ptr, ptr + (n_layers * this->config.dim)};
    this->weights.rms_ffn_weight = rms_ffn_weight;
    ptr += n_layers * this->config.dim;
    std::vector<float> w1{ptr, ptr + (n_layers * this->config.dim * this->config.hidden_dim)};
    this->weights.w1 = w1;
    ptr += n_layers * this->config.dim * this->config.hidden_dim;
    std::vector<float> w2{ptr, ptr + (n_layers * this->config.hidden_dim * this->config.dim)};
    this->weights.w2 = w2;
    ptr += n_layers * this->config.hidden_dim * this->config.dim;
    std::vector<float> w3{ptr, ptr + (n_layers * this->config.dim * this->config.hidden_dim)};
    this->weights.w3 = w3;
    ptr += n_layers * this->config.dim * this->config.hidden_dim;
    std::vector<float> rms_final_weight{ptr, ptr + this->config.dim};
    this->weights.rms_final_weight = rms_final_weight;
    ptr += this->config.dim;
    ptr += this->config.seq_len * head_size / 2;
    ptr += this->config.seq_len * head_size / 2;
    this->weights.wcls = this->weights.token_embedding_table;
}

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

void Transformer::ReadCheckPoint(const char* checkpoint) {
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