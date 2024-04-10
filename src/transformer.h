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
    std::vector<float> token_embedding_table;  // [vocab_size, dim]
    // weights for rmsnorms
    std::vector<float> rms_att_weight;  // [layer, dim]
    std::vector<float> rms_ffn_weight;  // [layer, dim]
    // weights for matmuls. note dim == n_heads * head_size
    std::vector<float> wq;  // [layer, dim, n_heads * head_size]
    std::vector<float> wk;  // [layer, dim, n_kv_heads * head_size]
    std::vector<float> wv;  // [layer, dim, n_kv_heads * head_size]
    std::vector<float> wo;  // [layer, n_heads * head_size, dim]
    // weights for ffn
    std::vector<float> w1;  // [layer, hidden_dim, dim]
    std::vector<float> w2;  // [layer, dim, hidden_dim]
    std::vector<float> w3;  // [layer, hidden_dim, dim]
    // final rmsnorm
    std::vector<float> rms_final_weight;  // [dim]
    // (optional) classifier weights for the logits, on the last layer
    std::vector<float> wcls; 
};

struct RunState {
    // current wave of activations
    std::vector<float> x;  // activation at current time stamp [dim]
    std::vector<float> xb;  // same, but inside a residual branch [dim]
    std::vector<float> xb2;  // an additional buffer just for convenience [dim]
    std::vector<float> hb;  // buffer for hidden dimension in the ffn [hidden_dim]
    std::vector<float> hb2;  // another buffer for hidden dimension in the ffn [hidden_dim]
    std::vector<float> q;  // query [dim]
    std::vector<float> k;  // key [dim]
    std::vector<float> v;  // value [dim]
    std::vector<float> att;  // buffer for scores/attention values [n_heads, seq_len]
    std::vector<float> logits;  // buffer for logits
    // kv cache
    std::vector<float> key_cache;  // [layer, seq_len, dim]
    std::vector<float> value_cache;  // [layer, seq_len, dim]
};

class Transformer
{
public:
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