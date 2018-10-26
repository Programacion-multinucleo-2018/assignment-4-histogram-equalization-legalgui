// SYSTEM LIBRARIES
#include <iostream>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <chrono>

// OPEN CV LIRBARIES
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define BIN_COUNT 256

void equalize_sequential_CPU(const unsigned char *data, unsigned char *output, const long size, const unsigned int *lookup){
  // ITERATE THROUGH ALL PIXELS
  for(int i = 0; i < size; i++){
      output[i] = lookup[data[i]];
  }

}

// HISTOGRAM SEQUENTIAL CPU
void histogram_sequential_CPU(unsigned int *output, const unsigned char *data, const int size){
  // CALCULATE FREQUENCY OF EACH VALUE
  for(int i = 0; i < size; i++)
    output[(int)data[i]]++;
}


void calculate_cfd(unsigned int *output, unsigned int *data, int size){
  long sum = 0;
  for(int i = 0; i < size; i++){
    // CALCULATE THE CUMULATIVE SUM
    if(data[i]) sum += data[i];
    output[i] = sum;
  }
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

// PRINT ARRAY
inline void print_array(unsigned int * arr, int size){
  for(int i = 0; i < size; i++)
    std::cout << arr[i] << std::endl;
}

// FUNCTION TO SUMMON THE KERNEL
void histogram_equalizer(const cv::Mat& input, cv::Mat& output){

  // INPUT.STEP GETS THE NUMBER OF BYTES FOR EACH ROW
	std::cout << "Input image step: " << input.step << " rows: " << input.rows << " cols: " << input.cols << std::endl;

  unsigned int image_histogram[256] = {0};
  histogram_sequential_CPU(image_histogram, input.ptr(), input.cols*input.rows);

  // GET THE NEW TABLE
  unsigned int cfd_histogram[256] = {0};
  calculate_cfd(cfd_histogram, image_histogram, 256);

  // TABULATE NEW VALUES
  unsigned int equalized_histogram[256] = {0};
  calculate_equalization(equalized_histogram, cfd_histogram, 256, input.cols*input.rows);

  equalize_sequential_CPU(input.data, output.data, input.cols*input.rows, equalized_histogram);
  std::cout << "Image equalized" << std::endl;
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
  auto start = std::chrono::high_resolution_clock::now();
  histogram_equalizer(input, output);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> duration_ms = end - start;
  printf("time seq (ms): %f\n", duration_ms.count());

  cv::imwrite("output.jpg", output);
  return 0;
}
