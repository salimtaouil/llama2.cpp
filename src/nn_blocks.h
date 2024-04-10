#ifndef NN_BLOCKS_H_
#define NN_BLOCKS_H_

#include "transformer.h"
#include <vector>

void rmsnorm(tensor1d &output, tensor1d &input, tensor1d &weight);
void softmax(tensor1d &output, tensor1d &input, int max_pos);
void matmul(tensor1d &output, tensor1d &input, tensor2d &weight);
void forward(int token_index, int token_position, Config &config, RunState &state, TransformerWeights &transformer_weights);

#endif
