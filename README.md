# Assignment 4: Histogram Equalization

Assignment No 4 for the multi-core programming course. Implement histogram equalization for a gray scale image in CPU and GPU. The result of applying the algorithm to an image with low contrast can be seen in Figure 1:

![Figure 1](Images/histogram_equalization.png)
<br/>Figure 1: Expected Result.

The programs have to do the following:

1. Using Opencv, load and image and convert it to grayscale.
2. Calculate de histogram of the image.
3. Calculate the normalized sum of the histogram.
4. Create an output image based on the normalized histogram.
5. Display both the input and output images.

Test your code with the different images that are included in the *Images* folder. Include the average calculation time for both the CPU and GPU versions, as well as the speedup obtained, in the Readme.

Rubric:

1. Image is loaded correctly.
2. The histogram is calculated correctly using atomic operations.
3. The normalized histogram is correctly calculated.
4. The output image is correctly calculated.
5. For the GPU version, used shared memory where necessary.
6. Both images are displayed at the end.
7. Calculation times and speedup obtained are incuded in the Readme.

GPU Time: For the GPU Time calculus, only the kernels execution times were measured. The reason for this is that memory access for this implemention is basically constant, while the computation complexities of the equalization kernels are more interesting. Measuring the whole process would be counter-productive because the execution times are so fast that memory allocation would basically slow down the process relatively a lot. Of course, this slow-down becomes marginal as the size of the picture, color bin and channels grow, but in this scenario, it absorbs all the speed-up performance.

CPU Average Time for dog3.jpeg and 256 bin: 88.115311 (ms)
GPU Average Time for dog3.jpeg and 256 bin: 0.044217 (ms) 4 kernels. 
Speed Up: x1992.7926
