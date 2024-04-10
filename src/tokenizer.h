#ifndef TOKENIZER_H_
#define TOKENIZER_H_

#include <string>
#include <vector>
#include <fstream>
#include <iostream>

std::vector<std::string> build_tokenizer(const char* tokenizer_path, int vocab_size);

#endif