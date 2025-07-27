#include <cuda_fp16.h>
#include <cudnn.h>
#include <cub/cub.cuh>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <cmath>

// Optimized version using warp primitives
template<int BLOCK_SIZE>
__global__ void softmax_argmax_kernel_optimized(
    const half* __restrict__ input,
    half* __restrict__ output,
    int* __restrict__ argmax_indices,
    const int vocab_size,
    const int seq_len) {
    
    const int seq_pos = blockIdx.x;
    if (seq_pos >= seq_len) return;
    
    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    const int num_warps = BLOCK_SIZE / 32;
    
    // Shared memory
    extern __shared__ char shared_mem_bytes[];
    float* warp_max = (float*)shared_mem_bytes;
    int* warp_max_idx = (int*)(warp_max + num_warps);
    float* warp_sum = (float*)(warp_max_idx + num_warps);
    
    // Phase 1: Find max
    float thread_max = -INFINITY;
    int thread_max_idx = 0;
    
    for (int vocab_idx = tid; vocab_idx < vocab_size; vocab_idx += BLOCK_SIZE) {
        float val = __half2float(input[seq_pos * vocab_size + vocab_idx]);
        if (val > thread_max) {
            thread_max = val;
            thread_max_idx = vocab_idx;
        }
    }
    
    // Warp-level reduction for max
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_max = __shfl_down_sync(0xffffffff, thread_max, offset);
        int other_idx = __shfl_down_sync(0xffffffff, thread_max_idx, offset);
        
        if (other_max > thread_max || (other_max == thread_max && other_idx < thread_max_idx)) {
            thread_max = other_max;
            thread_max_idx = other_idx;
        }
    }
    
    // Store warp results
    if (lane_id == 0) {
        warp_max[warp_id] = thread_max;
        warp_max_idx[warp_id] = thread_max_idx;
    }
    __syncthreads();
    
    // Final reduction across warps
    if (tid < num_warps) {
        thread_max = warp_max[tid];
        thread_max_idx = warp_max_idx[tid];
    } else {
        thread_max = -INFINITY;
        thread_max_idx = INT_MAX;
    }
    
    if (warp_id == 0) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            float other_max = __shfl_down_sync(0xffffffff, thread_max, offset);
            int other_idx = __shfl_down_sync(0xffffffff, thread_max_idx, offset);
            
            if (other_max > thread_max || (other_max == thread_max && other_idx < thread_max_idx)) {
                thread_max = other_max;
                thread_max_idx = other_idx;
            }
        }
        
        if (lane_id == 0) {
            warp_max[0] = thread_max;
            warp_max_idx[0] = thread_max_idx;
        }
    }
    __syncthreads();
    
    float max_val = warp_max[0];
    int max_idx = warp_max_idx[0];
    
    // Phase 2: Compute exp and sum
    float thread_sum = 0.0f;
    
    for (int vocab_idx = tid; vocab_idx < vocab_size; vocab_idx += BLOCK_SIZE) {
        float val = __half2float(input[seq_pos * vocab_size + vocab_idx]);
        float exp_val = expf(val - max_val);
        thread_sum += exp_val;
        output[seq_pos * vocab_size + vocab_idx] = __float2half(exp_val);
    }
    
    // Warp-level reduction for sum
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    
    if (lane_id == 0) {
        warp_sum[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // Final reduction
    if (tid < num_warps) {
        thread_sum = warp_sum[tid];
    } else {
        thread_sum = 0.0f;
    }
    
    if (warp_id == 0) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
        }
        
        if (lane_id == 0) {
            warp_sum[0] = thread_sum;
        }
    }
    __syncthreads();
    
    float total_sum = warp_sum[0];
    float inv_sum = 1.0f / total_sum;
    
    // Phase 3: Normalize
    for (int vocab_idx = tid; vocab_idx < vocab_size; vocab_idx += BLOCK_SIZE) {
        float exp_val = __half2float(output[seq_pos * vocab_size + vocab_idx]);
        output[seq_pos * vocab_size + vocab_idx] = __float2half(exp_val * inv_sum);
    }
    
    // Store argmax
    if (tid == 0) {
        argmax_indices[seq_pos] = max_idx;
    }
}

// Wrapper function
void launch_softmax_argmax(
    const half* input,
    half* output,
    int* argmax_indices,
    int vocab_size,
    int seq_len,
    cudaStream_t stream = 0) {
    
    const int block_size = 256;
    const int grid_size = seq_len;  // One block per sequence position
    
    
    // Use optimized kernel for larger vocabularies
    size_t shared_mem_size_opt = (block_size / 32) * sizeof(float) * 2 + (block_size / 32) * sizeof(int);
    softmax_argmax_kernel_optimized<block_size><<<grid_size, block_size, shared_mem_size_opt, stream>>>(
        input, output, argmax_indices, vocab_size, seq_len
    );
    
}

// Helper macro for checking CUDA errors
#define CHECK_CUDA(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " code=" << error << " \"" << cudaGetErrorString(error) << "\"" << std::endl; \
        exit(1); \
    } \
} while(0)

// Helper macro for checking cuDNN errors
#define CHECK_CUDNN(call) do { \
    cudnnStatus_t status = call; \
    if (status != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "cuDNN error at " << __FILE__ << ":" << __LINE__ \
                  << " code=" << status << " \"" << cudnnGetErrorString(status) << "\"" << std::endl; \
        exit(1); \
    } \
} while(0)

// CPU reference implementation for validation
void cpu_softmax_argmax(const half* input, half* output, int* argmax, 
                       int vocab_size, int seq_len) {
    std::vector<float> temp(vocab_size);
    
    for (int col = 0; col < seq_len; ++col) {
        // Find max
        float max_val = -INFINITY;
        int max_idx = 0;
        for (int row = 0; row < vocab_size; ++row) {
            float val = __half2float(input[col * vocab_size + row]);
            if (val > max_val) {
                max_val = val;
                max_idx = row;
            }
        }
        argmax[col] = max_idx;
        
        // Compute exp(x - max) and sum
        float sum = 0.0f;
        for (int row = 0; row < vocab_size; ++row) {
            float val = __half2float(input[col * vocab_size + row]);
            temp[row] = expf(val - max_val);
            sum += temp[row];
        }
        
        // Normalize
        for (int row = 0; row < vocab_size; ++row) {
            output[col * vocab_size + row] = __float2half(temp[row] / sum);
        }
    }
}

// Kernel to find argmax from softmax output (for cuDNN comparison)
__global__ void find_argmax_kernel(const half* softmax_output, int* argmax, 
                                  int vocab_size, int seq_len) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= seq_len) return;
    
    float max_val = -INFINITY;
    int max_idx = 0;
    
    for (int row = 0; row < vocab_size; ++row) {
        float val = __half2float(softmax_output[col * vocab_size + row]);
        if (val > max_val) {
            max_val = val;
            max_idx = row;
        }
    }
    
    argmax[col] = max_idx;
}

// Benchmark class
class SoftmaxBenchmark {
private:
    cudnnHandle_t cudnn_handle;
    cudnnTensorDescriptor_t input_desc, output_desc;
    
    // Device memory
    half *d_input, *d_output_custom, *d_output_cudnn;
    int *d_argmax_custom, *d_argmax_cudnn;
    
    // Host memory
    std::vector<half> h_input;
    std::vector<half> h_output_custom, h_output_cudnn, h_output_cpu;
    std::vector<int> h_argmax_custom, h_argmax_cudnn, h_argmax_cpu;
    
    int vocab_size;
    int seq_len;
    
public:
    SoftmaxBenchmark(int vocab_size, int seq_len) 
        : vocab_size(vocab_size), seq_len(seq_len) {
        
        // Initialize cuDNN
        CHECK_CUDNN(cudnnCreate(&cudnn_handle));
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));
        
        // Set tensor descriptors for column-major layout
        // cuDNN expects NCHW format, we'll use N=seq_len, C=1, H=vocab_size, W=1
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(
            input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF,
            seq_len, vocab_size, 1, 1
        ));
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(
            output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF,
            seq_len, vocab_size, 1, 1
        ));
        
        // Allocate memory
        size_t data_size = vocab_size * seq_len * sizeof(half);
        size_t argmax_size = seq_len * sizeof(int);
        
        CHECK_CUDA(cudaMalloc(&d_input, data_size));
        CHECK_CUDA(cudaMalloc(&d_output_custom, data_size));
        CHECK_CUDA(cudaMalloc(&d_output_cudnn, data_size));
        CHECK_CUDA(cudaMalloc(&d_argmax_custom, argmax_size));
        CHECK_CUDA(cudaMalloc(&d_argmax_cudnn, argmax_size));
        
        // Allocate host memory
        h_input.resize(vocab_size * seq_len);
        h_output_custom.resize(vocab_size * seq_len);
        h_output_cudnn.resize(vocab_size * seq_len);
        h_output_cpu.resize(vocab_size * seq_len);
        h_argmax_custom.resize(seq_len);
        h_argmax_cudnn.resize(seq_len);
        h_argmax_cpu.resize(seq_len);
        
        // Initialize input with random data
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        for (int i = 0; i < vocab_size * seq_len; ++i) {
            h_input[i] = __float2half(dist(gen));
        }
        
        // Copy input to device
        CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), data_size, cudaMemcpyHostToDevice));
    }
    
    ~SoftmaxBenchmark() {
        cudaFree(d_input);
        cudaFree(d_output_custom);
        cudaFree(d_output_cudnn);
        cudaFree(d_argmax_custom);
        cudaFree(d_argmax_cudnn);
        
        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroyTensorDescriptor(output_desc);
        cudnnDestroy(cudnn_handle);
    }
    
    void run_custom_kernel(int warmup_iters = 10, int bench_iters = 100) {
        // Warmup
        for (int i = 0; i < warmup_iters; ++i) {
            launch_softmax_argmax(d_input, d_output_custom, d_argmax_custom, 
                                vocab_size, seq_len);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // Benchmark
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < bench_iters; ++i) {
            launch_softmax_argmax(d_input, d_output_custom, d_argmax_custom, 
                                vocab_size, seq_len);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        
        std::cout << "Custom Kernel Performance:" << std::endl;
        std::cout << "  Average time: " << milliseconds / bench_iters << " ms" << std::endl;
        std::cout << "  Throughput: " << (vocab_size * seq_len * sizeof(half) * 2 * bench_iters) 
                  / (milliseconds * 1e6) << " GB/s" << std::endl;
        
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        
        // Copy results back
        CHECK_CUDA(cudaMemcpy(h_output_custom.data(), d_output_custom, 
                            vocab_size * seq_len * sizeof(half), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_argmax_custom.data(), d_argmax_custom, 
                            seq_len * sizeof(int), cudaMemcpyDeviceToHost));
    }
    
    void run_cudnn(int warmup_iters = 10, int bench_iters = 100) {
        float alpha = 1.0f, beta = 0.0f;
        
        // Warmup
        for (int i = 0; i < warmup_iters; ++i) {
            CHECK_CUDNN(cudnnSoftmaxForward(
                cudnn_handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
                &alpha, input_desc, d_input,
                &beta, output_desc, d_output_cudnn
            ));
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // Benchmark
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < bench_iters; ++i) {
            CHECK_CUDNN(cudnnSoftmaxForward(
                cudnn_handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
                &alpha, input_desc, d_input,
                &beta, output_desc, d_output_cudnn
            ));
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        
        std::cout << "\ncuDNN Performance:" << std::endl;
        std::cout << "  Average time: " << milliseconds / bench_iters << " ms" << std::endl;
        std::cout << "  Throughput: " << (vocab_size * seq_len * sizeof(half) * 2 * bench_iters) 
                  / (milliseconds * 1e6) << " GB/s" << std::endl;
        
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        
        // Find argmax for cuDNN output
        int block_size = 256;
        int grid_size = (seq_len + block_size - 1) / block_size;
        find_argmax_kernel<<<grid_size, block_size>>>(
            d_output_cudnn, d_argmax_cudnn, vocab_size, seq_len
        );
        
        // Copy results back
        CHECK_CUDA(cudaMemcpy(h_output_cudnn.data(), d_output_cudnn, 
                            vocab_size * seq_len * sizeof(half), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_argmax_cudnn.data(), d_argmax_cudnn, 
                            seq_len * sizeof(int), cudaMemcpyDeviceToHost));
    }
    
    void validate_results() {
        std::cout << "\nValidation Results:" << std::endl;
        
        // Run CPU reference
        cpu_softmax_argmax(h_input.data(), h_output_cpu.data(), h_argmax_cpu.data(), 
                          vocab_size, seq_len);
        
        // Compare softmax outputs
        float max_diff_custom = 0.0f;
        float max_diff_cudnn = 0.0f;
        float avg_diff_custom = 0.0f;
        float avg_diff_cudnn = 0.0f;
        
        for (int i = 0; i < vocab_size * seq_len; ++i) {
            float cpu_val = __half2float(h_output_cpu[i]);
            float custom_val = __half2float(h_output_custom[i]);
            float cudnn_val = __half2float(h_output_cudnn[i]);
            
            float diff_custom = std::abs(custom_val - cpu_val);
            float diff_cudnn = std::abs(cudnn_val - cpu_val);
            
            max_diff_custom = std::max(max_diff_custom, diff_custom);
            max_diff_cudnn = std::max(max_diff_cudnn, diff_cudnn);
            avg_diff_custom += diff_custom;
            avg_diff_cudnn += diff_cudnn;
        }
        
        avg_diff_custom /= (vocab_size * seq_len);
        avg_diff_cudnn /= (vocab_size * seq_len);
        
        // Compare custom vs cuDNN
        float max_diff_custom_cudnn = 0.0f;
        float avg_diff_custom_cudnn = 0.0f;
        
        for (int i = 0; i < vocab_size * seq_len; ++i) {
            float custom_val = __half2float(h_output_custom[i]);
            float cudnn_val = __half2float(h_output_cudnn[i]);
            
            float diff = std::abs(custom_val - cudnn_val);
            max_diff_custom_cudnn = std::max(max_diff_custom_cudnn, diff);
            avg_diff_custom_cudnn += diff;
        }
        avg_diff_custom_cudnn /= (vocab_size * seq_len);
        
        std::cout << "  Custom vs CPU - Max diff: " << max_diff_custom 
                  << ", Avg diff: " << avg_diff_custom << std::endl;
        std::cout << "  cuDNN vs CPU - Max diff: " << max_diff_cudnn 
                  << ", Avg diff: " << avg_diff_cudnn << std::endl;
        std::cout << "  Custom vs cuDNN - Max diff: " << max_diff_custom_cudnn 
                  << ", Avg diff: " << avg_diff_custom_cudnn << std::endl;
        
        // Compare argmax
        int argmax_mismatches_custom = 0;
        int argmax_mismatches_cudnn = 0;
        int argmax_mismatches_custom_cudnn = 0;
        
        for (int i = 0; i < seq_len; ++i) {
            if (h_argmax_custom[i] != h_argmax_cpu[i]) argmax_mismatches_custom++;
            if (h_argmax_cudnn[i] != h_argmax_cpu[i]) argmax_mismatches_cudnn++;
            if (h_argmax_custom[i] != h_argmax_cudnn[i]) argmax_mismatches_custom_cudnn++;
        }
        
        std::cout << "  Argmax mismatches - Custom vs CPU: " << argmax_mismatches_custom 
                  << "/" << seq_len << std::endl;
        std::cout << "  Argmax mismatches - cuDNN vs CPU: " << argmax_mismatches_cudnn 
                  << "/" << seq_len << std::endl;
        std::cout << "  Argmax mismatches - Custom vs cuDNN: " << argmax_mismatches_custom_cudnn 
                  << "/" << seq_len << std::endl;
        
        // Print sample outputs for inspection
        std::cout << "\nSample outputs (first 5 values of first column):" << std::endl;
        for (int i = 0; i < std::min(5, vocab_size); ++i) {
            std::cout << "  [" << i << "] CPU: " << __half2float(h_output_cpu[i])
                      << ", Custom: " << __half2float(h_output_custom[i])
                      << ", cuDNN: " << __half2float(h_output_cudnn[i]) << std::endl;
        }
        std::cout << "  Argmax[0] - CPU: " << h_argmax_cpu[0] 
                  << ", Custom: " << h_argmax_custom[0]
                  << ", cuDNN: " << h_argmax_cudnn[0] << std::endl;
    }
};

int main(int argc, char** argv) {
    // Test configurations
    std::vector<std::pair<int, int>> test_configs = {
        {128256, 1024},   // GPT-2 vocab size, moderate sequence
        {128256, 8192},   // GPT-2 vocab size, max sequence
        {151936, 1024},   // Larger vocab size, moderate sequence
        {151936, 8192}    // Larger vocab size, max sequence
    };
    
    for (const auto& config : test_configs) {
        int vocab_size = config.first;
        int seq_len = config.second;
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "Testing vocab_size=" << vocab_size 
                  << ", seq_len=" << seq_len << std::endl;
        std::cout << "========================================" << std::endl;
        
        try {
            SoftmaxBenchmark bench(vocab_size, seq_len);
            bench.run_custom_kernel();
            bench.run_cudnn();
            bench.validate_results();
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }
    
    return 0;
}