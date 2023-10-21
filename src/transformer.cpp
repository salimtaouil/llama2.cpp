#include "transformer.h"
#include <cstdio>
#include <iostream>
#include <cmath>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <memory>

Transformer::Transformer(const char* checkpoint)
{
    ReadCheckPoint(checkpoint);

    resize_run_state();
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