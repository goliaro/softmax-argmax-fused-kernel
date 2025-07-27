# softmax-argmax-fused-kernel
A simple Softmax/ArgMax fused CUDA kernel


Compile with: `nvcc -O3 -arch=sm_80 -std=c++14 softmax.cu -lcudnn -o softmax`
Run with: `./softmax`

