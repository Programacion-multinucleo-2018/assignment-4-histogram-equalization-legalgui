// OPEN CV LIRBARIES
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// CUDA LIBRARY
#include <cuda_runtime.h>

// CUDA CUSTOM LIBRARY
#include "common.h"
#include "helper.hpp"
#define BIN_COUNT 256
auto t = std::chrono::high_resolution_clock::now();
std::chrono::duration<float, std::milli> global_duration = t-t;

// HISTOGRAM COUNT GPU
__global__ void histogram_count_parallel_GPU(unsigned int *output, const unsigned char *input, const long color_bin){

  // CREATE SHARED SUB_HISTOGRAM
  __shared__ unsigned int sub_histogram[256];
  sub_histogram[threadIdx.x] = 0;
  __syncthreads();

  // GET THE XINDEX AND THE OFFSET
  int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
  int offset = blockDim.x * gridDim.x;

  // USE HANDLER
  int tid = threadIdx.x;

  while (xIndex < color_bin){
    // COUNT THE SUB_HISTOGRAM
    atomicAdd(&sub_histogram[input[xIndex]], 1);
    xIndex += offset;
  }
  __syncthreads();
  // MERGE THE SUB_HISTOGRAMS
  atomicAdd(&(output[tid]), sub_histogram[tid]);
}

// HISTOGRAM CFD PARALLEL GPU
__global__ void histogram_cfd_parallel_GPU(unsigned int *output, const unsigned int *input, const long color_bin){
  int xIndex = blockIdx.x;
  int id = 0;
  long sum = 0;

  if(xIndex < 256){
    while(id <= xIndex){
      sum += input[id];
      id++;
    }
    output[xIndex] = sum;
  }
}

// HISTOGRAM EQUALIZATION PARALLEL GPU
__global__ void histogram_equalization_parallel_GPU(unsigned int *output, const unsigned int *input, int color_bin, int min, int pixels){
  int xIndex = blockIdx.x;

  if(xIndex < 256){
    // CALCULATE THE EQUALIZED VALUE
    input[xIndex] ? output[xIndex] = lroundf(((float)(input[xIndex] - min)/(float)(pixels-1))*(color_bin-1)) :
    output[xIndex] = 0;
  }
}

// LOOKUP REPLACEMENT PARALLEL GPU
__global__ void lookup_replace_parallel_GPU(unsigned char *output, const unsigned char *input, const int color_bin, const unsigned int *lookup_table){

  // GET THE XINDEX AND THE OFFSET
	int xIndex = threadIdx.x + blockDim.x * blockIdx.x;
  int offset = blockDim.x * gridDim.x;

	while (xIndex < color_bin){
    // USE THE LOOKUP FUNCTION
    output[xIndex] = lookup_table[input[xIndex]];
    xIndex += offset;
  }
}


// GET THE CUDA PROPS
int get_cuda_device_props(){
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  return prop.multiProcessorCount;
}


// KERNEL WRAPPER FUNCTIONS
void run_histogram_count_kernel(int blocks, unsigned int *output, const unsigned char *input, int color_bin){
  auto start_cpu = std::chrono::high_resolution_clock::now();
  histogram_count_parallel_GPU<<<blocks*2, 256>>>(output, input, color_bin);
  auto end_cpu =  std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
  global_duration += duration_ms;
  printf("Calculating the picture histogram %f ms\n", duration_ms.count());
}

void run_histogram_cfd_kernel(int blocks, unsigned int *output, const unsigned int *input, int color_bin){
  auto start_cpu = std::chrono::high_resolution_clock::now();
  histogram_cfd_parallel_GPU<<<256, 1>>>(output, input, color_bin);
  auto end_cpu =  std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
  global_duration += duration_ms;
  printf("Calculating the histogram's cfd %f ms\n", duration_ms.count());
}

void run_histogram_equalization_kernel(int blocks, unsigned int *output, const unsigned int *input, int color_bin, int min, int pixels){
  auto start_cpu = std::chrono::high_resolution_clock::now();
  histogram_equalization_parallel_GPU<<<256, 1>>>(output, input, color_bin, min, pixels);
  auto end_cpu =  std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
  global_duration += duration_ms;
  printf("Equalizing the histogram %f ms\n", duration_ms.count());
}

void run_lookup_replace_kernel(int blocks, unsigned char *output, const unsigned char *input, int pixels, unsigned int *lookup_table){
  auto start_cpu = std::chrono::high_resolution_clock::now();
  lookup_replace_parallel_GPU<<<blocks*2,256>>>(output, input, pixels, lookup_table);
  auto end_cpu =  std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
  global_duration += duration_ms;
  printf("Processing the image %f ms\n", duration_ms.count());
}


// FUNCTION TO SUMMON THE KERNEL
void histogram_equalizer(const cv::Mat& input, cv::Mat& output){

  // INPUT.STEP GETS THE NUMBER OF BYTES FOR EACH ROW
	std::cout << "Input image step: " << input.step << " rows: " << input.rows << " cols: " << input.cols << std::endl;

  // SETUP
  // GET DEVICE PROPS
  int blocks = get_cuda_device_props();

  // CALCULATE TOTAL NUMBER OF BYTES OF INPUT AND OUTPUT IMAGE
  unsigned char *d_input;
  unsigned char *d_output;
  unsigned int *d_histogram_count;
  unsigned int *d_histogram_cfd;
  unsigned int *d_histogram_equalized_lookup;

  size_t input_output_bytes = input.step * input.rows;
	size_t histogram_bytes = 256 * sizeof(unsigned int);

  // ALLOCATE DEVICE MEMORY FOR FIRST INPUT AND FINAL OUTPUT
	SAFE_CALL(cudaMalloc<unsigned char>(&d_input, input_output_bytes), "CUDA Malloc Failed");
  SAFE_CALL(cudaMalloc<unsigned char>(&d_output, input_output_bytes), "CUDA Malloc Failed");

  SAFE_CALL(cudaMemset(d_output, 0, input_output_bytes), "CUDA Malloc Failed");

  // ALLOCATE MEMORY FOR HISTOGRAM COUNT AND HISTOGRAM EQUALIZED AND INITIALIZE
	SAFE_CALL(cudaMalloc<unsigned int>(&d_histogram_count, histogram_bytes), "CUDA Malloc Failed");
  SAFE_CALL(cudaMalloc<unsigned int>(&d_histogram_cfd, histogram_bytes), "CUDA Malloc Failed");
  SAFE_CALL(cudaMalloc<unsigned int>(&d_histogram_equalized_lookup, histogram_bytes), "CUDA Malloc Failed");

  SAFE_CALL(cudaMemset(d_histogram_count, 0, histogram_bytes), "CUDA Memset Failed");
  SAFE_CALL(cudaMemset(d_histogram_cfd, 0, histogram_bytes), "CUDA Memset Failed");
  SAFE_CALL(cudaMemset(d_histogram_equalized_lookup, 0, histogram_bytes), "CUDA Memset Failed");

  // COPY DATA FROM OPENCV INPUT IMAGE TO DEVICE MEMORY
	SAFE_CALL(cudaMemcpy(d_input, input.ptr(), input_output_bytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");


  // BEGIN
  // EXECUTE KERNEL
  run_histogram_count_kernel(blocks, d_histogram_count, d_input, input.cols*input.rows);

  // COPY HISTOGRAM TO HOST MEMORY
  unsigned int image_histogram[256];
  SAFE_CALL(cudaMemcpy(image_histogram, d_histogram_count, histogram_bytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

  // CASE GPU CALCULATION WAS CORRECT
  if(assert_gpu_vs_cpu(input.ptr(), image_histogram, input.cols*input.rows)){
    std::cout << "GPU histogram calculation was successful" << std::endl;

    // GET THE NEW TABLE
    // unsigned int cfd_histogram_count[256] = {0};
    // calculate_cfd(cfd_histogram_count, image_histogram, 256);
    //
    // // COPY CFD HISTOGRAM TO DEVICE
    // SAFE_CALL(cudaMemcpy(d_histogram_cfd, cfd_histogram_count, histogram_bytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");
    //

    run_histogram_cfd_kernel(1, d_histogram_cfd, d_histogram_count, 256);
    // RUN KERNEL
    // int min = get_min_nonzero_value(cfd_histogram_count, 256);
    int min = 11;
    run_histogram_equalization_kernel(1, d_histogram_equalized_lookup, d_histogram_cfd, 256, min, input.cols*input.rows);

    //EXECUTE KERNEL
    run_lookup_replace_kernel(blocks, d_output, d_input, input.cols * input.rows, d_histogram_equalized_lookup);

    // COPY BACK DATA
	  SAFE_CALL(cudaMemcpy(output.ptr(), d_output, input_output_bytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");
    printf("Total Time %f ms\n", global_duration.count());
  }else{ // CASE IT WASN'T
    std::cout << "Histogram calculation found an error, closing software" << std::endl;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_histogram_count);
    cudaFree(d_histogram_cfd);
    cudaFree(d_histogram_equalized_lookup);
    exit(1);
  }
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_histogram_count);
  cudaFree(d_histogram_cfd);
  cudaFree(d_histogram_equalized_lookup);
  return;
}

int main(int argc, char *argv[]) {
  // GET THE IMAGE PATH
  std::string imagePath;
  (argc < 2) ? imagePath = "images/scenery.jpg" : imagePath = argv[1];

  // READ INPUT IMAGE FROM DISK
  cv::Mat input, colorInput = cv::imread(imagePath, CV_LOAD_IMAGE_COLOR);

  if (colorInput.empty()){
    std::cout << "Image Not Found!" << std::endl;
    std::cin.get();
    return -1;
  }

  // GET THE IMAGE AND CONVERT IT TO GRAYSCALE
  cv::cvtColor(colorInput, input, CV_BGR2GRAY);

  // CREATE OUTPUT IMAGE
	cv::Mat output = input.clone();

  // CALL THE WRAPPER FUNCTION
  histogram_equalizer(input, output);

  cv::imwrite("output.jpg", output);
  return 0;
}
