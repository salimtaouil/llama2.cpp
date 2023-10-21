#include <iostream>
#include "transformer.h"

int main(){

    Transformer t("stories15M.bin");


    std::cout << "Config dim: " << t.config.dim << "\n";
    std::cout << "Config hidden_dim: " << t.config.hidden_dim << "\n";
    std::cout << "Config n_layers: " << t.config.n_layers << "\n";
    std::cout << "Config n_heads: " << t.config.n_heads << "\n";
    std::cout << "Config n_kv_heads: " << t.config.n_kv_heads << "\n";
    std::cout << "Config size: " << t.config.vocab_size << "\n";
    std::cout << "Config seq_len: " << t.config.seq_len << "\n";
    std::cout << "weights: " << t.weights.token_embedding_table[100] << "\n";
    
    std::cout << "state.x: " << t.state.x.size() << "\n";
    return 0;
}