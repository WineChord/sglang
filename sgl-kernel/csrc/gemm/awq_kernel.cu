// Adapted from
// https://github.com/vllm-project/vllm/blob/eb59b5a6cba6727d3727c0372258db9002f687c1/csrc/quantization/awq/gemm_kernels.cu#L350
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/all.h>

__device__ uint4 dequantize_s4_to_fp16x2(uint32_t const& source) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  uint4 result;

  uint32_t* h = reinterpret_cast<uint32_t*>(&result);
  uint32_t const i4s = reinterpret_cast<uint32_t const&>(source);

  // First, we extract the i4s and construct an intermediate fp16 number.
  static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
  static constexpr uint32_t BOTTOM_MASK = 0x000f000f;
  static constexpr uint32_t TOP_MASK = 0x00f000f0;
  static constexpr uint32_t I4s_TO_F16s_MAGIC_NUM = 0x64006400;

  // Note that the entire sequence only requires 1 shift instruction. This is
  // thanks to the register packing format and the fact that we force our
  // integers to be unsigned, and account for this in the fp16 subtractions. In
  // addition, I exploit the fact that sub and fma have the same throughput in
  // order to convert elt_23 and elt_67 to fp16 without having to shift them to
  // the bottom bits before hand.

  // Shift right by 8 to now consider elt_45 and elt_67. Issue first to hide RAW
  // dependency if we issue immediately before required.
  const uint32_t top_i4s = i4s >> 8;
  // Extract elt_01 - (i4s & 0x000f000f) | 0x64006400
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[0])
               : "r"(i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
  // Extract elt_23 (i4s & 0x00f000f0) | 0x64006400
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[1])
               : "r"(i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
  // Extract elt_45 (top_i4s & 0x000f000f) | 0x64006400
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[2])
               : "r"(top_i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
  // Extract elt_67 (top_i4s & 0x00f000f0) | 0x64006400
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[3])
               : "r"(top_i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));

  // This is the half2 {1024, 1024} represented as an integer.
  static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64006400;
  // This is the half2 {1 / 16, 1 / 16} represented as an integer.
  static constexpr uint32_t ONE_SIXTEENTH = 0x2c002c00;
  // This is the half2 {-64, -64} represented as an integer.
  static constexpr uint32_t NEG_64 = 0xd400d400;

  // Finally, we construct the output numbers.
  // Convert elt_01
  asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(FP16_TOP_MAGIC_NUM));
  // Convert elt_23
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[1]) : "r"(h[1]), "r"(ONE_SIXTEENTH), "r"(NEG_64));
  // Convert elt_45
  asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[2]) : "r"(h[2]), "r"(FP16_TOP_MAGIC_NUM));
  // Convert elt_67
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[3]) : "r"(h[3]), "r"(ONE_SIXTEENTH), "r"(NEG_64));

  return result;
#else
  assert(false);
  return {};
#endif
}

__device__ void dequantize_s4_to_bf16x2(uint32_t const& source, 
                                        __nv_bfloat162& result1, 
                                        __nv_bfloat162& result2, 
                                        __nv_bfloat162& result3, 
                                        __nv_bfloat162& result4) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  // Extract 4-bit values directly from the packed int.
  uint32_t const i4s = source;
  const uint32_t top_i4s = i4s >> 8;  // Separate the top bits as in original implementation.
  
  // Directly extract nibbles (4-bit values) from the 32-bit integer.
  // This matches the original byte layout pattern.
  uint32_t nibbles[8];
  
  // Extract each 4-bit value, correctly matching original layout.
  nibbles[0] = i4s & 0xF;                  // Bottom 4 bits of first byte.
  nibbles[1] = (i4s >> 4) & 0xF;           // Top 4 bits of first byte.
  nibbles[2] = (i4s >> 16) & 0xF;          // Bottom 4 bits of third byte.
  nibbles[3] = (i4s >> 20) & 0xF;          // Top 4 bits of third byte.
  nibbles[4] = top_i4s & 0xF;              // Bottom 4 bits of second byte.
  nibbles[5] = (top_i4s >> 4) & 0xF;       // Top 4 bits of second byte.  
  nibbles[6] = (top_i4s >> 16) & 0xF;      // Bottom 4 bits of fourth byte.
  nibbles[7] = (top_i4s >> 20) & 0xF;      // Top 4 bits of fourth byte.

  // Convert directly to BF16 via float.
  // Using constant values for faster math.
  const float bias = 64.0f;
  const float scale = 1.0f/16.0f;
  
  // First group transformations (subtract magic number equivalent).
  float values[8];
  
  // Process first elt_01 and elt_45 (as in original implementation).
  values[0] = static_cast<float>(nibbles[0]) - bias;
  values[1] = static_cast<float>(nibbles[1]) - bias;
  values[2] = static_cast<float>(nibbles[2]) - bias;
  values[3] = static_cast<float>(nibbles[3]) - bias;
  
  // Process elt_23 and elt_67 (applying scaling and bias as in original implementation).
  values[4] = static_cast<float>(nibbles[4]) * scale - bias;
  values[5] = static_cast<float>(nibbles[5]) * scale - bias;
  values[6] = static_cast<float>(nibbles[6]) * scale - bias;
  values[7] = static_cast<float>(nibbles[7]) * scale - bias;
  
  // Create bf16x2 pairs directly from float values.
  // This uses native bf16 instructions without any fp16 involvement.
  result1 = __floats2bfloat162_rn(values[0], values[1]);
  result2 = __floats2bfloat162_rn(values[2], values[3]);
  result3 = __floats2bfloat162_rn(values[4], values[5]);
  result4 = __floats2bfloat162_rn(values[6], values[7]);
#else
  // This code path should not be executed on older architectures.
  assert(false);
#endif
}

__global__ void __launch_bounds__(256) dequantize_weights_fp16(
    int* __restrict__ qweight,
    half* __restrict__ scales,
    int* __restrict__ qzeros,
    half* __restrict__ output,
    int group_size,
    int qweight_cols) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  uint4 zeros = dequantize_s4_to_fp16x2(qzeros[col + (row / group_size) * qweight_cols]);
  uint4 loaded_scale = *(uint4*)(scales + 8 * col + (row / group_size) * qweight_cols * 8);

  uint4 weight_fp16 = dequantize_s4_to_fp16x2(qweight[col + row * qweight_cols]);

  asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(weight_fp16.x) : "r"(weight_fp16.x), "r"(zeros.x));
  asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(weight_fp16.x) : "r"(weight_fp16.x), "r"(loaded_scale.x));
  asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(weight_fp16.y) : "r"(weight_fp16.y), "r"(zeros.y));
  asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(weight_fp16.y) : "r"(weight_fp16.y), "r"(loaded_scale.y));
  asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(weight_fp16.z) : "r"(weight_fp16.z), "r"(zeros.z));
  asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(weight_fp16.z) : "r"(weight_fp16.z), "r"(loaded_scale.z));
  asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(weight_fp16.w) : "r"(weight_fp16.w), "r"(zeros.w));
  asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(weight_fp16.w) : "r"(weight_fp16.w), "r"(loaded_scale.w));

  half* output_ptr = output + 8 * col + 8 * row * qweight_cols;
  *(uint4*)output_ptr = weight_fp16;
}

__global__ void __launch_bounds__(256) dequantize_weights_bf16(
    int* __restrict__ qweight,
    __nv_bfloat16* __restrict__ scales,
    int* __restrict__ qzeros,
    __nv_bfloat16* __restrict__ output,
    int group_size,
    int qweight_cols) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  
  // Get scales in BF16 format.
  __nv_bfloat162 scale_bf16_1 = *(__nv_bfloat162*)(scales + 8 * col + (row / group_size) * qweight_cols * 8);
  __nv_bfloat162 scale_bf16_2 = *(__nv_bfloat162*)(scales + 8 * col + (row / group_size) * qweight_cols * 8 + 2);
  __nv_bfloat162 scale_bf16_3 = *(__nv_bfloat162*)(scales + 8 * col + (row / group_size) * qweight_cols * 8 + 4);
  __nv_bfloat162 scale_bf16_4 = *(__nv_bfloat162*)(scales + 8 * col + (row / group_size) * qweight_cols * 8 + 6);
  
  // Convert zeros directly to BF16.
  __nv_bfloat162 zeros_bf16_1, zeros_bf16_2, zeros_bf16_3, zeros_bf16_4;
  dequantize_s4_to_bf16x2(qzeros[col + (row / group_size) * qweight_cols], 
                         zeros_bf16_1, zeros_bf16_2, zeros_bf16_3, zeros_bf16_4);
  
  // Convert weights directly to BF16.
  __nv_bfloat162 weight_bf16_1, weight_bf16_2, weight_bf16_3, weight_bf16_4;
  dequantize_s4_to_bf16x2(qweight[col + row * qweight_cols], 
                         weight_bf16_1, weight_bf16_2, weight_bf16_3, weight_bf16_4);
  
  // Dequantize the weights using native BF16 operations.
  weight_bf16_1 = __hsub2(weight_bf16_1, zeros_bf16_1);
  weight_bf16_1 = __hmul2(weight_bf16_1, scale_bf16_1);
  
  weight_bf16_2 = __hsub2(weight_bf16_2, zeros_bf16_2);
  weight_bf16_2 = __hmul2(weight_bf16_2, scale_bf16_2);
  
  weight_bf16_3 = __hsub2(weight_bf16_3, zeros_bf16_3);
  weight_bf16_3 = __hmul2(weight_bf16_3, scale_bf16_3);
  
  weight_bf16_4 = __hsub2(weight_bf16_4, zeros_bf16_4);
  weight_bf16_4 = __hmul2(weight_bf16_4, scale_bf16_4);
  
  // Store the results.
  __nv_bfloat16* output_ptr = output + 8 * col + 8 * row * qweight_cols;
  *(__nv_bfloat162*)(output_ptr) = weight_bf16_1;
  *(__nv_bfloat162*)(output_ptr + 2) = weight_bf16_2;
  *(__nv_bfloat162*)(output_ptr + 4) = weight_bf16_3;
  *(__nv_bfloat162*)(output_ptr + 6) = weight_bf16_4;
#endif
}

// Template function to handle multiple data types.
template<typename T>
torch::Tensor awq_dequantize_impl(torch::Tensor qweight, torch::Tensor scales, torch::Tensor qzeros) {
  int qweight_rows = qweight.size(0);
  int qweight_cols = qweight.size(1);
  int group_size = qweight_rows / scales.size(0);

  int x_num_threads = 16;
  int y_num_threads = 16;
  int x_blocks = qweight_cols / x_num_threads;
  int y_blocks = qweight_rows / y_num_threads;

  const at::cuda::OptionalCUDAGuard device_guard(device_of(qweight));
  
  // Use the template type T for the output tensor.
  auto output_tensor_options = torch::TensorOptions().dtype(scales.dtype()).device(scales.device());
  at::Tensor output = torch::empty({qweight_rows, qweight_cols * 8}, output_tensor_options);

  dim3 num_blocks(x_blocks, y_blocks);
  dim3 threads_per_block(x_num_threads, y_num_threads);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto _qweight = reinterpret_cast<int*>(qweight.data_ptr<int>());
  auto _scales = reinterpret_cast<T*>(scales.data_ptr<T>());
  auto _zeros = reinterpret_cast<int*>(qzeros.data_ptr<int>());
  auto _output = reinterpret_cast<T*>(output.data_ptr<T>());
  
  if constexpr (std::is_same_v<T, at::Half>) {
    dequantize_weights_fp16<<<num_blocks, threads_per_block, 0, stream>>>(
        _qweight, reinterpret_cast<half*>(_scales), _zeros, reinterpret_cast<half*>(_output), 
        group_size, qweight_cols);
  } else if constexpr (std::is_same_v<T, at::BFloat16>) {
    int device_idx;
    cudaGetDevice(&device_idx);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_idx);
    
    if (prop.major >= 8) {
      // GPU supports BF16 natively (Ampere or later).
      dequantize_weights_bf16<<<num_blocks, threads_per_block, 0, stream>>>(
          _qweight, reinterpret_cast<__nv_bfloat16*>(_scales), _zeros, 
          reinterpret_cast<__nv_bfloat16*>(_output), group_size, qweight_cols);
    } else {
      // For older GPUs that don't support BF16 natively, convert to FP16
      at::Tensor scales_fp16 = scales.to(torch::kFloat16);
      at::Tensor output_fp16 = torch::empty({qweight_rows, qweight_cols * 8}, 
          torch::TensorOptions().dtype(torch::kFloat16).device(scales.device()));
      
      auto _scales_fp16 = reinterpret_cast<half*>(scales_fp16.data_ptr<at::Half>());
      auto _output_fp16 = reinterpret_cast<half*>(output_fp16.data_ptr<at::Half>());
      
      dequantize_weights_fp16<<<num_blocks, threads_per_block, 0, stream>>>(
          _qweight, _scales_fp16, _zeros, _output_fp16, group_size, qweight_cols);
      
      // Convert back to BF16
      output = output_fp16.to(torch::kBFloat16);
    }
  }

  return output;
}

torch::Tensor awq_dequantize(torch::Tensor qweight, torch::Tensor scales, torch::Tensor qzeros) {
  if (scales.scalar_type() == torch::kFloat16) {
    return awq_dequantize_impl<at::Half>(qweight, scales, qzeros);
  } else if (scales.scalar_type() == torch::kBFloat16) {
    return awq_dequantize_impl<at::BFloat16>(qweight, scales, qzeros);
  } else {
    AT_ERROR("Unsupported data type for AWQ dequantization: ", scales.scalar_type());
  }
}

