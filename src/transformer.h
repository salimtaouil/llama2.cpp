#ifndef TRANSFORMER_H_
#define TRANSFORMER_H_

#include <vector>

typedef std::vector<float> tensor1d;
typedef std::vector<tensor1d> tensor2d;
typedef std::vector<tensor2d> tensor3d;

float EPS = 1e-5;

struct Config {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
};

struct TransformerWeights {
    tensor2d token_embedding_table;  // [vocab_size, dim]
    // weights for rmsnorms
    tensor2d rms_att_weight;  // [layer, dim]
    tensor2d rms_ffn_weight;  // [layer, dim]
    // weights for attention matmuls
    tensor3d wq;  // [layer, dim, dim]
    tensor3d wk;  // [layer, dim, dim]
    tensor3d wv;  // [layer, dim, dim]
    tensor3d wo;  // [layer, dim, dim]
    // weights for ffn
    tensor3d w1;  // [layer, hidden_dim, dim]
    tensor3d w2;  // [layer, dim, hidden_dim]
    tensor3d w3;  // [layer, hidden_dim, dim]
    // final rmsnorm
    tensor1d rms_final_weight;  // [dim]
    // freq_cis for RoPE relatively positional embeddings
    tensor2d freq_cis_real;  // [seq_len, (dim/n_heads)/2]
    tensor2d freq_cis_imag;  // [seq_len, (dim/n_heads)/2]
};

struct RunState {
    // current wave of activations
    tensor1d x;  // activation at current time stamp [dim]
    tensor1d xb;  // same, but inside a residual branch [dim]
    tensor1d xb2;  // an additional buffer just for convenience [dim]
    tensor1d hb;  // buffer for hidden dimension in the ffn [hidden_dim]
    tensor1d hb2;  // another buffer for hidden dimension in the ffn [hidden_dim]
    tensor1d q;  // query [dim]
    tensor1d k;  // key [dim]
    tensor1d v;  // value [dim]
    tensor1d attention;  // buffer for scores/attention values [seq_len]
    tensor1d logits;  // buffer for logits [vocab_size]
    // kv cache
    tensor3d key_cache;  // [layer, seq_len, dim]
    tensor3d value_cache;  // [layer, seq_len, dim]
};

class Transformer
{
public:
    /* data */
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    /* methods */
    Transformer(const char* checkpoint);
    ~Transformer();
    void ReadCheckPoint(const char* checkpoint);
    void resize_state_tensors();
    void resize_weights_tensors();
};

#endif