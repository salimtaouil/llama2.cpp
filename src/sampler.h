#ifndef SAMPLER_H_
#define SAMPLER_H_

#include "transformer.h"
#include <vector>
#include <cmath>
#include <random>

int sample(tensor1d &probabilities);
int argmax(tensor1d &values);

#endif