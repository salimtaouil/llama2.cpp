#include "transformer.h"
#include "nn_blocks.h"
#include "tokenizer.h"
#include "sampler.h"
#include <iostream>
#include <vector>
#include <string>
#include <time.h>

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// genetation loop
void generate(Transformer& transformer, int steps, std::vector<std::string> &vocab; float temperature){

    // start main loop
    long start = 0;
    int next;
    int token = 1;
    int pos = 0;
    while(pos < steps){
        // forward the transformer to get logits for the next token
        forward(token, pos, transformer.config, transformer.state, transformer.weights);

        // advance the state machine
        if(temperature < EPS){
            next = argmax(transformer.state.logits);
        }
        else {
            for (int q = 0; q < transformer.config.vocab_size; q++) transformer.state.logits[q] /= temperature;
            softmax(transformer.state.logits);
            next = sample(transformer.state.logits);
        }
        std::cout << vocab[next] << std::fflush;
        token = next;
        pos++;
        //init the timer here because the first iteration can be slower
        if (start == 0){ start = time_in_ms();}
    }
    std::cout << "\n";

    if(pos > 1){
        long end = time_in_ms();
        std::cout << "achieved tok/s: " << (pos-1) / (double)(end - start)*1000 << "\n";
    }
}

int main(){

    float temperature = 0.9f;
    int steps;

    Transformer transformer("model.bin");
    steps = transformer.config.seq_len;
  
    std::vector<std::string> vocab = build_tokenizer("tokenizer.bin", transformer.config.vocab_size);

    generate(transformer, steps, vocab, temperature);
    
    return 0;
}