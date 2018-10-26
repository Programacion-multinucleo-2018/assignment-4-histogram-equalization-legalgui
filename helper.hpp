// SYSTEM LIBRARIES
#include <iostream>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <numeric>

// PRINT ARRAY
inline void print_array(unsigned int * arr, int size){
  for(int i = 0; i < size; i++)
    std::cout << arr[i] << std::endl;
}

// GET MINIMUM NONZERO VALUE
int get_min_nonzero_value(unsigned int *data, int size){

  int i = 0;
  // TRAVERSE TO THE FIRST NON-ZERO VALUE
  while(i < size && data[i] == 0) i++;

  int smallest = data[i];
  for(int i = 1;  i < size; ++i){
    if(data[i] < smallest && data[i] != 0)
      smallest = data[i];
  }
  return smallest;
}

int get_min_max_value(unsigned int *data, int size, std::string flag){
  if(flag == "max")
    // CALL GET MAX ELEMENT
    return *std::max_element(data, data + size);
  // CALL GET MIN NONZERO VALUE
  return get_min_nonzero_value(data, size);
}

void calculate_equalization(unsigned int *output, unsigned int *data, int size, int pixels){
  // GET MIN MAX VALUE
  int cdf_min = get_min_max_value(data, 256, "min"); // GET MIN VALUE
  int cdf_max = get_min_max_value(data, 256, "max"); // GET MAX VALUE NOT USED
  for(int i = 0; i < 256; i++){
    // CALCULATE THE EQUALIZED VALUE
    data[i] ? output[i] = std::round(((float)(data[i] - cdf_min)/(float)(pixels-1))*(size-1)) :
    output[i] = 0;
  }

}

void calculate_cfd(unsigned int *output, unsigned int *data, int size){
  long sum = 0;
  for(int i = 0; i < size; i++){
    // CALCULATE THE CUMULATIVE SUM
    if(data[i]) sum += data[i];
    output[i] = sum;
  }
}

// HISTOGRAM SEQUENTIAL CPU
void histogram_sequential_CPU(unsigned int *output, const unsigned char *data, const int size){
  // CALCULATE FREQUENCY OF EACH VALUE
  for(int i = 0; i < size; i++)
    output[(int)data[i]]++;
}

void check_histogram_CPU(const int sum, const int size){
  // SUM MUST BE EQUAL TO SIZE
  sum == size ? std::cout << "CPU Frequencies seem ok" << std::endl :
  std::cout << "Something went wrong with the CPU frequency calculation" << std::endl;
}

// ASSERT CORRECT CALCULUS
bool assert_gpu_vs_cpu(const unsigned char *data, const unsigned int *gpu_histogram, const int size){

  // GET SEQUENTIAL HISTOGRAM CALCULATION
  unsigned int cpu_histogram[256] = {0};
  histogram_sequential_CPU(cpu_histogram, data, size);

  // VERIFY GPU AND CPU TOTAL SUM
  long gpu_sum = 0, cpu_sum = 0;
  gpu_sum += std::accumulate(gpu_histogram, gpu_histogram + 256, 0);
  cpu_sum += std::accumulate(cpu_histogram, cpu_histogram + 256, 0);

  check_histogram_CPU(cpu_sum, size);

  // CHECK IF CALCULATIONS MATCH
  if(cpu_sum == gpu_sum){
    std::cout << "CPU and GPU calculations match" << std::endl;

    // CREATE TEMPORAL AUXILIARY HISTOGRAM
    unsigned int aux_gpu_histogram[256];
    memcpy(aux_gpu_histogram, gpu_histogram, 256 * sizeof(int));

    // VERIFY IF HISTOGRAM IS REVERSIBLE
    for (int i = 0; i < size; i++)
      aux_gpu_histogram[data[i]]--;

    // VERIFY IF HISTOGRAM IS BACK TO ZERO
    for(int i=0; i < 256; i++){
      if(aux_gpu_histogram[i]){
        std::cout << "GPU Histogram was not reversible" << std::endl;
        return 0;
      }
    }

    std::cout << "GPU Histogram was reversible" << std::endl;
    return 1;
  }
  std::cout << "CPU and GPU calculations don't match" << std::endl;
  return 0;
}


void equalize_sequential_CPU(const unsigned char *data, unsigned char *output, const long size, const unsigned int *lookup){
  // ITERATE THROUGH ALL PIXELS
  for(int i = 0; i < size; i++)
    output[i] = lookup[data[i]];
}

