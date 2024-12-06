
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <opencv2/opencv.hpp>

#include <stdio.h>
#include <iostream>

#define BLOCK_SIZE 32
#define WIN_SIZE 3
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

// kernel to get the Ix, Iy and It of two images
__global__ void getDerivatives(int* Ix, int* Iy, int* It, int* I1, int* I2, int width, int height)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int idx = i + j * width;
	if (i > 0 && i < width - 1 && j > 0 && j < height - 1)
	{
		Ix[idx] = (I1[idx + 1] - I1[idx - 1] + I2[idx + 1] - I2[idx - 1]) / 4;
		Iy[idx] = (I1[idx + width] - I1[idx - width] + I2[idx + width] - I2[idx - width]) / 4;
		It[idx] = (I2[idx] - I1[idx]);
	}
}

//kernel to get Ix, Iy and It of two images using shared memory
__global__ void getDerivatives_2(int* Ix, int* Iy, int* It, int* I1, int* I2, int width, int height)
{	
	__shared__ int I1_shared[BLOCK_SIZE + 2][BLOCK_SIZE + 2];
	__shared__ int I2_shared[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int idx = i + j * width;

	I1_shared[threadIdx.y + 1][threadIdx.x + 1] = I1[idx];
	I2_shared[threadIdx.y + 1][threadIdx.x + 1] = I2[idx];


	if (threadIdx.x == 0 && i > 0)
	{
		I1_shared[threadIdx.y + 1][threadIdx.x] = I1[idx - 1];
		I2_shared[threadIdx.y + 1][threadIdx.x] = I2[idx - 1];
	}

	if (threadIdx.x == blockDim.x - 1 && i < width - 1)
	{
		I1_shared[threadIdx.y + 1][threadIdx.x + 2] = I1[idx + 1];
		I2_shared[threadIdx.y + 1][threadIdx.x + 2] = I2[idx + 1];
	}

	if (threadIdx.y == 0 && j > 0)
	{
		I1_shared[threadIdx.y][threadIdx.x + 1] = I1[idx - width];
		I2_shared[threadIdx.y][threadIdx.x + 1] = I2[idx - width];
	}

	if (threadIdx.y == blockDim.y - 1 && j < height - 1)
	{
		I1_shared[threadIdx.y + 2][threadIdx.x + 1] = I1[idx + width];
		I2_shared[threadIdx.y + 2][threadIdx.x + 1] = I2[idx + width];
	}

	// Wait for all threads to finish copying
	__syncthreads();

	if (i > 0 && i < width - 1 && j > 0 && j < height - 1) {
		Ix[idx] = (I1_shared[threadIdx.y + 1][threadIdx.x + 2] - I1_shared[threadIdx.y + 1][threadIdx.x] + I2_shared[threadIdx.y + 1][threadIdx.x + 2] - I2_shared[threadIdx.y + 1][threadIdx.x]) / 4;
		Iy[idx] = (I1_shared[threadIdx.y + 2][threadIdx.x + 1] - I1_shared[threadIdx.y][threadIdx.x + 1] + I2_shared[threadIdx.y + 2][threadIdx.x + 1] - I2_shared[threadIdx.y][threadIdx.x + 1]) / 4;
		It[idx] = (I2_shared[threadIdx.y + 1][threadIdx.x + 1] - I1_shared[threadIdx.y + 1][threadIdx.x + 1]);
	}
}

__global__ void computeOpticalFlow_GPU(int* Ix, int* Iy, int* It, int width, int height, int stride, int* u, int* v)
{

	__shared__ int Ix_shared[BLOCK_SIZE + WIN_SIZE - 1][BLOCK_SIZE + WIN_SIZE - 1];
	__shared__ int Iy_shared[BLOCK_SIZE + WIN_SIZE - 1][BLOCK_SIZE + WIN_SIZE - 1];
	__shared__ int It_shared[BLOCK_SIZE + WIN_SIZE - 1][BLOCK_SIZE + WIN_SIZE - 1];

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int idx = y * stride + x;

	Ix_shared[threadIdx.y + WIN_SIZE / 2][threadIdx.x + WIN_SIZE / 2] = Ix[idx];
	Iy_shared[threadIdx.y + WIN_SIZE / 2][threadIdx.x + WIN_SIZE / 2] = Iy[idx];
	It_shared[threadIdx.y + WIN_SIZE / 2][threadIdx.x + WIN_SIZE / 2] = It[idx];

	if (threadIdx.x < WIN_SIZE / 2 && x > 0)
	{
		Ix_shared[threadIdx.y + WIN_SIZE / 2][threadIdx.x] = Ix[idx - WIN_SIZE / 2];
		Iy_shared[threadIdx.y + WIN_SIZE / 2][threadIdx.x] = Iy[idx - WIN_SIZE / 2];
		It_shared[threadIdx.y + WIN_SIZE / 2][threadIdx.x] = It[idx - WIN_SIZE / 2];
	}

	if (threadIdx.x >= blockDim.x - WIN_SIZE / 2 && x < width - 1)
	{
		Ix_shared[threadIdx.y + WIN_SIZE / 2][threadIdx.x + WIN_SIZE - 1] = Ix[idx + WIN_SIZE / 2];
		Iy_shared[threadIdx.y + WIN_SIZE / 2][threadIdx.x + WIN_SIZE - 1] = Iy[idx + WIN_SIZE / 2];
		It_shared[threadIdx.y + WIN_SIZE / 2][threadIdx.x + WIN_SIZE - 1] = It[idx + WIN_SIZE / 2];
	}

	if (threadIdx.y < WIN_SIZE / 2 && y > 0)
	{
		Ix_shared[threadIdx.y][threadIdx.x + WIN_SIZE / 2] = Ix[idx - WIN_SIZE / 2 * stride];
		Iy_shared[threadIdx.y][threadIdx.x + WIN_SIZE / 2] = Iy[idx - WIN_SIZE / 2 * stride];
		It_shared[threadIdx.y][threadIdx.x + WIN_SIZE / 2] = It[idx - WIN_SIZE / 2 * stride];
	}

	if (threadIdx.y >= blockDim.y - WIN_SIZE / 2 && y < height - 1)
	{
		Ix_shared[threadIdx.y + WIN_SIZE - 1][threadIdx.x + WIN_SIZE / 2] = Ix[idx + WIN_SIZE / 2 * stride];
		Iy_shared[threadIdx.y + WIN_SIZE - 1][threadIdx.x + WIN_SIZE / 2] = Iy[idx + WIN_SIZE / 2 * stride];
		It_shared[threadIdx.y + WIN_SIZE - 1][threadIdx.x + WIN_SIZE / 2] = It[idx + WIN_SIZE / 2 * stride];
	}

	__syncthreads();
	if (x < WIN_SIZE / 2 || x >= width - WIN_SIZE / 2 || y < WIN_SIZE / 2 || y >= height - WIN_SIZE / 2)
	{
		u[y * stride + x] = 0;
		v[y * stride + x] = 0;
		return;
	}

	float sumIx2 = 0, sumIy2 = 0, sumIxIy = 0, sumIxIt = 0, sumIyIt = 0;
	// Compute the sums
	for (int wy = -WIN_SIZE / 2; wy <= WIN_SIZE / 2; wy++)
	{
		for (int wx = -WIN_SIZE / 2; wx <= WIN_SIZE / 2; wx++)
		{
			sumIx2 += Ix_shared[threadIdx.y + wy + WIN_SIZE / 2][threadIdx.x + wx + WIN_SIZE / 2] * Ix_shared[threadIdx.y + wy + WIN_SIZE / 2][threadIdx.x + wx + WIN_SIZE / 2];
			sumIy2 += Iy_shared[threadIdx.y + wy + WIN_SIZE / 2][threadIdx.x + wx + WIN_SIZE / 2] * Iy_shared[threadIdx.y + wy + WIN_SIZE / 2][threadIdx.x + wx + WIN_SIZE / 2];
			sumIxIy += Ix_shared[threadIdx.y + wy + WIN_SIZE / 2][threadIdx.x + wx + WIN_SIZE / 2] * Iy_shared[threadIdx.y + wy + WIN_SIZE / 2][threadIdx.x + wx + WIN_SIZE / 2];
			sumIxIt += Ix_shared[threadIdx.y + wy + WIN_SIZE / 2][threadIdx.x + wx + WIN_SIZE / 2] * It_shared[threadIdx.y + wy + WIN_SIZE / 2][threadIdx.x + wx + WIN_SIZE / 2];
			sumIyIt += Iy_shared[threadIdx.y + wy + WIN_SIZE / 2][threadIdx.x + wx + WIN_SIZE / 2] * It_shared[threadIdx.y + wy + WIN_SIZE / 2][threadIdx.x + wx + WIN_SIZE / 2];
		}
	}

	float det = sumIx2 * sumIy2 - sumIxIy * sumIxIy;
	if (fabs(det) > 1e-6) {
		u[y * stride + x] = (sumIy2 * -sumIxIt - sumIxIy * -sumIyIt) / det;
		v[y * stride + x] = (sumIx2 * -sumIyIt - sumIxIy * -sumIxIt) / det;
	}
	else {
		u[y * stride + x] = 0;
		v[y * stride + x] = 0;
	}
}

// Compute image gradients using Sobel operator
void computeGradients(int* gray, int width, int height, int stride, int* Ix, int* Iy, int* It, int* prevGray) {
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int idx = y * stride + x;

            Ix[idx] = (gray[idx + 1] - gray[idx - 1] + prevGray[idx + 1] - prevGray[idx - 1]) / 4.0f;
            Iy[idx] = (gray[idx + stride] - gray[idx - stride] + prevGray[idx + stride] - prevGray[idx - stride]) / 4.0f;
            It[idx] = (float)(gray[idx] - prevGray[idx]);
        }
    }
}


// Lucas-Kanade method
void computeOpticalFlow(int* Ix, int* Iy, int* It, int width, int height, int stride, int* u, int* v) {
	for (int y = WIN_SIZE / 2; y < height - WIN_SIZE / 2; y++) {
		for (int x = WIN_SIZE / 2; x < width - WIN_SIZE / 2; x++) {
			float sumIx2 = 0, sumIy2 = 0, sumIxIy = 0, sumIxIt = 0, sumIyIt = 0;

			for (int wy = -WIN_SIZE / 2; wy <= WIN_SIZE / 2; wy++) {
				for (int wx = -WIN_SIZE / 2; wx <= WIN_SIZE / 2; wx++) {
					int idx = (y + wy) * stride + (x + wx);
					sumIx2 += Ix[idx] * Ix[idx];
					sumIy2 += Iy[idx] * Iy[idx];
					sumIxIy += Ix[idx] * Iy[idx];
					sumIxIt += Ix[idx] * It[idx];
					sumIyIt += Iy[idx] * It[idx];
				}
			}

			float det = sumIx2 * sumIy2 - sumIxIy * sumIxIy;
			if (fabs(det) > 1e-6) {
				u[y * stride + x] = (sumIy2 * -sumIxIt - sumIxIy * -sumIyIt) / det;
				v[y * stride + x] = (sumIx2 * -sumIyIt - sumIxIy * -sumIxIt) / det;
			}
			else {
				u[y * stride + x] = 0;
				v[y * stride + x] = 0;
			}
		}
	}
}

int main()
{
    // Open video
    cv::VideoCapture cap("C:/Users/anand/Downloads/vehicles.mp4");
    if (!cap.isOpened()) {
        std::printf("Error: Unable to open video file.\n");
        return -1;
    }

    int width = 512;
    int height = 512;
    std::cout << width << " " << height << std::endl;
    int stride = width;

	// Allocate gpu host memory for the images
	int* I1 = new int[width * height];
	int* I2 = new int[width * height];
	int* Ix = new int[width * height];
	int* Iy = new int[width * height];
	int* It = new int[width * height];
	int* u = new int[width * height];
	int* v = new int[width * height];

	// initialize to zero
	for (int i = 0; i < width * height; i++) {
		I1[i] = 0;
		I2[i] = 0;
		Ix[i] = 0;
		Iy[i] = 0;
		It[i] = 0;
		u[i] = 0;
		v[i] = 0;
	}

	// Read the first frame
	cv::Mat frame;
	cap >> frame;
	cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
	cv::resize(frame, frame, cv::Size(width, height));
    for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			I1[i + j * stride] = frame.at<uchar>(j, i);
		}
    }

	// Read the second frame
	cap >> frame;
	cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
	cv::resize(frame, frame, cv::Size(width, height));
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			I2[i + j * stride] = frame.at<uchar>(j, i);
		}
	}

	// Allocate gpu memory for the images
	int* d_I1, * d_I2, * d_Ix, * d_Iy, * d_It, * d_u, * d_v;
	cudaMalloc(&d_I1, width * height * sizeof(int));
	cudaMalloc(&d_I2, width * height * sizeof(int));
	cudaMalloc(&d_Ix, width * height * sizeof(int));
	cudaMalloc(&d_Iy, width * height * sizeof(int));
	cudaMalloc(&d_It, width * height * sizeof(int));
	cudaMalloc(&d_u, width * height * sizeof(int));
	cudaMalloc(&d_v, width * height * sizeof(int));

	// Copy the images to the gpu memory
	cudaMemcpy(d_I1, I1, width * height * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_I2, I2, width * height * sizeof(int), cudaMemcpyHostToDevice);
    
	// Launch the kernel to get the derivatives
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);
	getDerivatives_2<<<numBlocks, threadsPerBlock>>> (d_Ix, d_Iy, d_It, d_I1, d_I2, width, height);

	// Launch the kernel to compute the optical flow
	computeOpticalFlow_GPU << <numBlocks, threadsPerBlock >> > (d_Ix, d_Iy, d_It, width, height, stride, d_u, d_v);

	// Copy the derivatives back to the host memory
	cudaMemcpy(Ix, d_Ix, width * height * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(Iy, d_Iy, width * height * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(It, d_It, width * height * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(u, d_u, width * height * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(v, d_v, width * height * sizeof(int), cudaMemcpyDeviceToHost);

	// Free the gpu memory
	cudaFree(d_I1);
	cudaFree(d_I2);
	cudaFree(d_Ix);
	cudaFree(d_Iy);
	cudaFree(d_It);
	cudaFree(d_u);
	cudaFree(d_v);

	//cpu code
	int* Ix_cpu = new int[width * height];
	int* Iy_cpu = new int[width * height];
	int* It_cpu = new int[width * height];
	int* u_cpu = new int[width * height];
	int* v_cpu = new int[width * height];

	//initialize to zero
	for (int i = 0; i < width * height; i++) {
		Ix_cpu[i] = 0;
		Iy_cpu[i] = 0;
		It_cpu[i] = 0;
		u_cpu[i] = 0;
		v_cpu[i] = 0;
	}

	computeGradients(I2, width, height, stride, Ix_cpu, Iy_cpu, It_cpu, I1);
	computeOpticalFlow(Ix_cpu, Iy_cpu, It_cpu, width, height, stride, u_cpu, v_cpu);


	// Compare the results
	float error = 0;
	for (int i = 0; i < width * height; i++) {
		if (abs(Ix[i] - Ix_cpu[i]) > 0) {
			std::cout << "Ix:" << i << " " << Ix[i] << " " << Ix_cpu[i] << std::endl;
		}
		if (abs(Iy[i] - Iy_cpu[i]) > 0) {
			std::cout << "Iy:" << i << " " << Ix[i] << " " << Ix_cpu[i] << std::endl;
		}
		if (abs(It[i] - It_cpu[i]) > 0) {
			std::cout << "It:" << i << " " << It[i] << " " << It_cpu[i] << std::endl;
		}
		if (abs(v[i] - v_cpu[i]) > 0) {
			std::cout << "u:" << i << " " << u[i] << " " << u_cpu[i] << std::endl;
			std::cout << "v:" << i << " " << v[i] << " " << v_cpu[i] << std::endl;
		}
		error += abs(Ix[i] - Ix_cpu[i]) + abs(Iy[i] - Iy_cpu[i]) + abs(It[i] - It_cpu[i]) + abs(u[i] - u_cpu[i]) + abs(v[i] - v_cpu[i]);
	}
	std::cout << "Error: " << error << std::endl;

	// Free the host memory
	delete[] I1;
	delete[] I2;
	delete[] Ix;
	delete[] Iy;
	delete[] It;
	delete[] u;
	delete[] v;
	std::cout << "Done" << std::endl;
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
