#include "sampler.h"
#include "transformer.h"
#include <vector>
#include <cmath>
#include <random>

int sample(tensor1d &probabilities) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);
    float r = dis(gen);

    float cdf = 0.0;
    for (int i = 0; i < probabilities.size(); ++i) {
        cdf += probabilities[i];
        if (r < cdf)
            return i;
    }
    // in case of rounding errors
    return probabilities.size() - 1;
}

int argmax(tensor1d &values) {
    int max_i = 0;
    float max_value = values[0];
    for (int i = 1; i < values.size(); ++i)
        if (values[i] > max_value) {
            max_i = i;
            max_value = values[i];
        }
    return max_i;
}


