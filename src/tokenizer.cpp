#include "tokenizer.h"
#include <string>
#include <vector>
#include <fstream>
#include <iostream>

std::vector<std::string> build_tokenizer(const std::string &tokenizer_path, int vocab_size){
    std::vector<std::string> vocab(vocab_size);
    {
        std::fstream file(tokenizer_path);
        if (!file) {
            std::cout
                << "Unable to open the tokenizer file tokenizer.bin!\n";
        }
        for (int i = 0; i < vocab_size; i++) {
            int len;
            vocab[i] = "";
            file.read((char*)&len, sizeof(int));
            for (int j = 0; j < len; ++j) {
                char c;
                file.read((char*)&c, sizeof(char));
                vocab[i].push_back(c);
            }
            vocab[i].push_back('\0');
        }
        file.close();
    }

    return vocab;
}