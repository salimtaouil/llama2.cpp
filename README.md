# CPPND: Capstone llama2.cpp

This repo includes the Capstone project in the [Udacity C++ Nanodegree Program](https://www.udacity.com/course/c-plus-plus-nanodegree--nd213). The code was developed in order to satisfy all criteria for the “README” and “Compiling and Testing” sections in [Rubric](https://review.udacity.com/#!/rubrics/2533/view), and at least 5 total criteria from the rest of the specification.

<img src="Run.png"/>

## The Capstone project (llama2.cpp) meets following project requirements:

### A. Introduction:
This is a C++ port of the awesome [llama2.c](https://github.com/karpathy/llama2.c/tree/master).

The program inference a baby [Llama 2](https://ai.meta.com/llama/) model in pure C++ and generate a new TinyStories. The model was trained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset.

Code Structure:

1- transformer.h/transformer.cpp: defines the transformer model (Config, Weights and RunState) also allows to read a model checkpoint from a .bin file.

2- nn_blocks.h/nn_blocks.cpp: defines the neural net blocks and the dynamics of the transformer. 

3- tokenizer.h/tokenize.cpp: Reads the tokenizer.bin file to a vector of strings and return a vocab vector.

4- sampler.h/sampler.cpp: defines the sampling functions that calculates the next token based on the calculated probabilities of the next word.

5- main.cpp: defines the generate loop.

### B. Criterias which support the Capstone Project:

1. Loops, Functions --> A variety of control structures are added to the project (transformer.h). The project code is clearly organized into functions.
2. I/O --> Application read from files (ReadCheckPoint(), build_tokenizer())

3. Class constructors utilize member initialization lists --> transformer.h

4. One or more classes are added to the project with appropriate access specifiers for class members --> Transformer class in transfomer.h

5. The project uses data structures and immutable variables. --> The project uses vectors (transformer.h) and uses constant variables (main.cpp).
6. Overloaded functions allow the same function to operate on different parameters. --> One function is overloaded with different signatures for the same function name. (checkpoint_init_tensor() in transformer.cpp)
7. The project makes use of references in function declarations- --> At least two functions use pass-by-reference in the project code (nn_blocks.cpp)
8. The project uses move semantics to move data instead of copying it, where possible --> in main.cpp the transformer object and vocab vector are moved to the generation thread.
9. The project uses multithreading --> main.cpp
10. A mutex or lock is used in the project --> std::unique_lock is used in the genartion loop.
11. The project uses scope / Resource Acquisition Is Initialization (RAII) where appropriate --> main.cpp


## Dependencies for Running Locally
* cmake >= 3.7
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Make a build directory in the top level directory: `mkdir build && cd build`
2. Compile: `cmake .. && make`
3. Switch to parent dir: `cd ..`
3. Run it: `build/run`.
