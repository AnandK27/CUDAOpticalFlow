#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
//#include <device_functions.h>
using namespace std;

#define WIN_SIZE 5  // Window size for Lucas-Kanade method
#define BLOCK_SIZE 32

#define DISPLAY_STREAMS false
#define CPU false
#define DEBUG false

#define CHECK(call) \
{ \
    cudaError_t error = call;                   \
    if (error != cudaSuccess) \
    { \
        printf("Error in File: %s, Line: %d\n", __FILE__, __LINE__); \
        printf("Error: %s\n", cudaGetErrorString(error)); \
        exit(1); \
    } \
} \

struct Gradients
{   
    float Ix;
    float Iy;
    float It;
};

/********************************
*
*   CPU Code
*
*********************************/
// Compute image gradients using Sobel operator
void computeGradients(const unsigned char* I1, int width, int height, int stride, float* Ix, float* Iy, float* It, const unsigned char* I2) {
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int idx = y * stride + x;

            Ix[idx] = (I1[idx + 1] - I1[idx - 1] + I2[idx + 1] - I2[idx - 1]) / 4.0f;
            Iy[idx] = (I1[idx + stride] - I1[idx - stride] + I2[idx + stride] - I2[idx - stride]) / 4.0f;
            It[idx] = (float)(I1[idx] - I2[idx]);
        }
    }
}

// Lucas-Kanade method
void computeOpticalFlow(const float* Ix, const float* Iy, const float* It, int width, int height, int stride, float* u, float* v) {
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

/***************************************
*
*   GPU Code
*
****************************************/

// kernel to get the Ix, Iy and It of two images
__global__ void cudaComputeGradients(float* Ix, float* Iy, float* It, const unsigned char* I1, const unsigned char* I2, int width, int height)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = i + j * width;
    if (i > 0 && i < width - 1 && j > 0 && j < height - 1)
    {
        Ix[idx] = (I1[idx + 1] - I1[idx - 1] + I2[idx + 1] - I2[idx - 1]) / 4.0f;
        Iy[idx] = (I1[idx + width] - I1[idx - width] + I2[idx + width] - I2[idx - width]) / 4.0f;
        It[idx] = (I2[idx] - I1[idx]);
    }
}

//kernel to get Ix, Iy and It of two images using shared memory
__global__ void cudaComputeGradients2(Gradients* grad, const unsigned char* I1, const unsigned char* I2, int width, int height)
{

    __shared__ int I1_shared[BLOCK_SIZE + 2][BLOCK_SIZE + 2];
    __shared__ int I2_shared[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = x + y * width;

    I1_shared[threadIdx.y + 1][threadIdx.x + 1] = I1[idx];
    I2_shared[threadIdx.y + 1][threadIdx.x + 1] = I2[idx];


    if (threadIdx.x == 0 && x > 0)
    {
        I1_shared[threadIdx.y + 1][threadIdx.x] = I1[idx - 1];
        I2_shared[threadIdx.y + 1][threadIdx.x] = I2[idx - 1];
    }

    if (threadIdx.x == blockDim.x - 1 && x < width - 1)
    {
        I1_shared[threadIdx.y + 1][threadIdx.x + 2] = I1[idx + 1];
        I2_shared[threadIdx.y + 1][threadIdx.x + 2] = I2[idx + 1];
    }

    if (threadIdx.y == 0 && y > 0)
    {
        I1_shared[threadIdx.y][threadIdx.x + 1] = I1[idx - width];
        I2_shared[threadIdx.y][threadIdx.x + 1] = I2[idx - width];
    }

    if (threadIdx.y == blockDim.y - 1 && y < height - 1)
    {
        I1_shared[threadIdx.y + 2][threadIdx.x + 1] = I1[idx + width];
        I2_shared[threadIdx.y + 2][threadIdx.x + 1] = I2[idx + width];
    }

    // Wait for all threads to finish copying
    __syncthreads();

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        Gradients* temp = &grad[idx];
        temp->Ix = (I1_shared[threadIdx.y + 1][threadIdx.x + 2] - I1_shared[threadIdx.y + 1][threadIdx.x] + I2_shared[threadIdx.y + 1][threadIdx.x + 2] - I2_shared[threadIdx.y + 1][threadIdx.x]) / 4.0f;
        temp->Iy = (I1_shared[threadIdx.y + 2][threadIdx.x + 1] - I1_shared[threadIdx.y][threadIdx.x + 1] + I2_shared[threadIdx.y + 2][threadIdx.x + 1] - I2_shared[threadIdx.y][threadIdx.x + 1]) / 4.0f;
        temp->It = (I2_shared[threadIdx.y + 1][threadIdx.x + 1] - I1_shared[threadIdx.y + 1][threadIdx.x + 1]);
    }
}


__global__ void cudaComputeOpticalFlow(const float* Ix, const float* Iy, const float* It, int width, int height, int stride, float* u, float* v)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y + (WIN_SIZE / 2);
    int x = blockIdx.x * blockDim.x + threadIdx.x + (WIN_SIZE / 2);

    if (x >= (width - WIN_SIZE / 2) || y >= (height - WIN_SIZE / 2)) {
        return;
    }

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

__global__ void cudaComputeOpticalFlow2(const Gradients* grad, int width, int height, int stride, float* u, float* v)
{
    __shared__ float Ix_shared[BLOCK_SIZE + WIN_SIZE - 1][BLOCK_SIZE + WIN_SIZE - 1];
    __shared__ float Iy_shared[BLOCK_SIZE + WIN_SIZE - 1][BLOCK_SIZE + WIN_SIZE - 1];
    __shared__ float It_shared[BLOCK_SIZE + WIN_SIZE - 1][BLOCK_SIZE + WIN_SIZE - 1];

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int idx = y * stride + x;

    if (y < height && x < width)
    {
        Gradients temp = grad[idx];
        Ix_shared[threadIdx.y + WIN_SIZE / 2][threadIdx.x + WIN_SIZE / 2] = temp.Ix;
        Iy_shared[threadIdx.y + WIN_SIZE / 2][threadIdx.x + WIN_SIZE / 2] = temp.Iy;
        It_shared[threadIdx.y + WIN_SIZE / 2][threadIdx.x + WIN_SIZE / 2] = temp.It;
    }

    // Left Edge
    if (threadIdx.x < WIN_SIZE / 2 && x >= WIN_SIZE / 2)
    {
        Gradients temp = grad[idx - WIN_SIZE / 2];
        Ix_shared[threadIdx.y + WIN_SIZE / 2][threadIdx.x] = temp.Ix;
        Iy_shared[threadIdx.y + WIN_SIZE / 2][threadIdx.x] = temp.Iy;
        It_shared[threadIdx.y + WIN_SIZE / 2][threadIdx.x] = temp.It;
    }

    // Right Edge
    if (threadIdx.x >= blockDim.x - WIN_SIZE / 2 && x < width - WIN_SIZE / 2)
    {
        Gradients temp = grad[idx + WIN_SIZE / 2];
        Ix_shared[threadIdx.y + WIN_SIZE / 2][threadIdx.x + WIN_SIZE - 1] = temp.Ix;
        Iy_shared[threadIdx.y + WIN_SIZE / 2][threadIdx.x + WIN_SIZE - 1] = temp.Iy;
        It_shared[threadIdx.y + WIN_SIZE / 2][threadIdx.x + WIN_SIZE - 1] = temp.It;
    }

    // Top Edge
    if (threadIdx.y < WIN_SIZE / 2 && y >= WIN_SIZE / 2)
    {
        Gradients temp = grad[idx - WIN_SIZE / 2 * stride];
        Ix_shared[threadIdx.y][threadIdx.x + WIN_SIZE / 2] = temp.Ix;
        Iy_shared[threadIdx.y][threadIdx.x + WIN_SIZE / 2] = temp.Iy;
        It_shared[threadIdx.y][threadIdx.x + WIN_SIZE / 2] = temp.It;
    }

    // Bottom Edge
    if (threadIdx.y >= blockDim.y - WIN_SIZE / 2 && y < height - WIN_SIZE / 2)
    {
        Gradients temp = grad[idx + WIN_SIZE / 2 * stride];
        Ix_shared[threadIdx.y + WIN_SIZE - 1][threadIdx.x + WIN_SIZE / 2] = temp.Ix;
        Iy_shared[threadIdx.y + WIN_SIZE - 1][threadIdx.x + WIN_SIZE / 2] = temp.Iy;
        It_shared[threadIdx.y + WIN_SIZE - 1][threadIdx.x + WIN_SIZE / 2] = temp.It;
    }

    // Top-Left Corner
    if (threadIdx.x == 0 && threadIdx.y == 0 && x > 0 && y > 0)
    {
        for (int wy = -WIN_SIZE / 2; wy < 0; wy++)
        {
            for (int wx = -WIN_SIZE / 2; wx < 0; wx++)
            {
                Gradients temp = grad[idx + (wy * stride) + wx];
                Ix_shared[threadIdx.y + WIN_SIZE / 2 + wy][threadIdx.x + WIN_SIZE / 2 + wx] = temp.Ix;
                Iy_shared[threadIdx.y + WIN_SIZE / 2 + wy][threadIdx.x + WIN_SIZE / 2 + wx] = temp.Iy;
                It_shared[threadIdx.y + WIN_SIZE / 2 + wy][threadIdx.x + WIN_SIZE / 2 + wx] = temp.It;
            }
        }
    }

    // Top-Right Corner
    if (threadIdx.x == (blockDim.x - 1) && threadIdx.y == 0 && x < (width - WIN_SIZE / 2) && y > 0)
    {
        for (int wy = -WIN_SIZE / 2; wy < 0; wy++)
        {
            for (int wx = 1; wx <= WIN_SIZE / 2; wx++)
            {
                Gradients temp = grad[idx + (wy * stride) + wx];
                Ix_shared[threadIdx.y + WIN_SIZE / 2 + wy][threadIdx.x + WIN_SIZE / 2 + wx] = temp.Ix;
                Iy_shared[threadIdx.y + WIN_SIZE / 2 + wy][threadIdx.x + WIN_SIZE / 2 + wx] = temp.Iy;
                It_shared[threadIdx.y + WIN_SIZE / 2 + wy][threadIdx.x + WIN_SIZE / 2 + wx] = temp.It;
            }
        }
    }

    // Bottom-Left
    if (threadIdx.x == 0 && threadIdx.y == (blockDim.y - 1) && x > 0 && y < (height - WIN_SIZE / 2))
    {
        for (int wy = 1; wy <= WIN_SIZE / 2; wy++)
        {
            for (int wx = -WIN_SIZE / 2; wx < 0; wx++)
            {
                Gradients temp = grad[idx + (wy * stride) + wx];
                Ix_shared[threadIdx.y + WIN_SIZE / 2 + wy][threadIdx.x + WIN_SIZE / 2 + wx] = temp.Ix;
                Iy_shared[threadIdx.y + WIN_SIZE / 2 + wy][threadIdx.x + WIN_SIZE / 2 + wx] = temp.Iy;
                It_shared[threadIdx.y + WIN_SIZE / 2 + wy][threadIdx.x + WIN_SIZE / 2 + wx] = temp.It;
            }
        }
    }

    // Bottom-Right
    if (threadIdx.x == (blockDim.x - 1) && threadIdx.y == (blockDim.y - 1) && x < (width - WIN_SIZE / 2) && y < (height - WIN_SIZE / 2))
    {
        for (int wy = 1; wy <= WIN_SIZE / 2; wy++)
        {
            for (int wx = 1; wx <= WIN_SIZE / 2; wx++)
            {
                Gradients temp = grad[idx + (wy * stride) + wx];
                Ix_shared[threadIdx.y + WIN_SIZE / 2 + wy][threadIdx.x + WIN_SIZE / 2 + wx] = temp.Ix;
                Iy_shared[threadIdx.y + WIN_SIZE / 2 + wy][threadIdx.x + WIN_SIZE / 2 + wx] = temp.Iy;
                It_shared[threadIdx.y + WIN_SIZE / 2 + wy][threadIdx.x + WIN_SIZE / 2 + wx] = temp.It;
            }
        }
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


/*****************************************
*
*   Utils
*
******************************************/

void compare(const float* a, const float* b, int width, int height, const char* str)
{
    printf("%s\n", str);
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int idx = y * width + x;
            if (abs(a[idx] - b[idx]) > 1e-3)
            {
                printf("Mismatch at idx: %d, %f, %f\n", idx, a[idx], b[idx]);
            }
        }
    }
}


// Visualize optical flow as HSV
void visualizeOpticalFlow(const float* u, const float* v, int width, int height, int stride, unsigned char* output) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * stride + x;

            float magnitude = sqrt(u[idx] * u[idx] + v[idx] * v[idx]);
            float angle = (float)atan2(v[idx], u[idx]) * 180.0f / CV_PI + 180.0f; // Convert to degrees

            float normMagnitude = fmin(magnitude / 10.0f, 1.0f); // Clipping magnitude

            // Convert HSV to RGB
            float h = angle / 2.0f; // [0, 6)
            float s = 0.5f;
            float v = normMagnitude;

            output[idx * 3 + 0] = (unsigned char)(h);
            output[idx * 3 + 1] = (unsigned char)(s * 255);
            output[idx * 3 + 2] = (unsigned char)(v * 255);
        }
    }
}

void writeToFile(const float* Ix, int width, int height, const char* file_name)
{
    FILE* f = fopen(file_name, "w");
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            fprintf(f, "%f,", Ix[i * width + j]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <video_file>\n", argv[0]);
        return -1;
    }

    // Open video
    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        printf("Error: Unable to open video file.\n");
        return -1;
    }

    int width = 512;
    int height = 512;
    int stride = width;
    int size = width * height * sizeof(float);

    // CPU Memory Allocation
    cv::Mat frame;
    unsigned char* temp = NULL;
    unsigned char* I1 = (unsigned char*)malloc(height * stride);
    unsigned char* I2 = (unsigned char*)malloc(height * stride);
    float* Ix = (float*)calloc(height * stride, sizeof(float));
    float* Iy = (float*)calloc(height * stride, sizeof(float));
    float* It = (float*)calloc(height * stride, sizeof(float));
    float* u = (float*)calloc(height * stride, sizeof(float));
    float* v = (float*)calloc(height * stride, sizeof(float));
    unsigned char* output = (unsigned char*)calloc(height * stride * 3, sizeof(unsigned char));

    // GPU Memory Allocation
    unsigned char* d_I1, * d_I2;
    Gradients* d_grad;
    float* d_u, * d_v;

    cudaMalloc(&d_I1, width * height);
    cudaMalloc(&d_I2, width * height);
    cudaMalloc(&d_grad, size * 3);
    cudaMalloc(&d_u, size);
    cudaMalloc(&d_v, size);

    cudaMemset(&d_grad, 0, size * 3);
    cudaMemset(&d_u, 0, size);
    cudaMemset(&d_v, 0, size);

    // Debug 
    Gradients* grad_cpu = (Gradients*)malloc(size * 3);
    float* Ix_cpu = (float*)malloc(size);
    float* Iy_cpu = (float*)malloc(size);
    float* It_cpu = (float*)malloc(size);
    float* u_cpu = (float*)malloc(size);
    float* v_cpu = (float*)malloc(size);

    time_t tick = time(NULL);
    float calc_fps = 0;
    int frame_num = 0;
    while (true)
    {
        int ret = cap.read(frame);
        if (!ret) break;
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        cv::resize(frame, frame, cv::Size(width, height));

#if DISPLAY_STREAMS
        cv::imshow("Input Frame Display", frame);
        double fps = cap.get(cv::CAP_PROP_FPS);
        int delay = static_cast<int>(1000 / fps); // Delay between frames in milliseconds

        if (cv::waitKey(delay) == 'q') {
            std::cout << "Exiting video playback" << std::endl;
            break;
        }
#endif

        frame_num++;
        temp = I1;
        I1 = I2;
        I2 = temp;

        temp = d_I1;
        d_I1 = d_I2;
        d_I2 = temp;

        memcpy(I2, frame.data, height * width * sizeof(unsigned char));
        cudaMemcpy(d_I2, frame.data, height * width * sizeof(unsigned char), cudaMemcpyHostToDevice);

#if CPU
        computeGradients(I2, width, height, stride, Ix, Iy, It, I1);
        computeOpticalFlow(Ix, Iy, It, width, height, stride, u, v);
        //visualizeOpticalFlow(u, v, width, height, stride, output);
#endif

        dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);
        cudaComputeGradients2 << <numBlocks, threadsPerBlock >> > (d_grad, d_I1, d_I2, width, height);

#if DEBUG
        cudaMemcpy(grad_cpu, d_grad, size * 3, cudaMemcpyDeviceToHost);
        for (int i = 0; i < width * height; i++)
        {
            Ix_cpu[i] = grad_cpu[i].Ix;
            Iy_cpu[i] = grad_cpu[i].Iy;
            It_cpu[i] = grad_cpu[i].It;
        }

        if (frame_num > 1)
        {
            compare(Ix, Ix_cpu, width, height, "Ix");
            compare(Iy, Iy_cpu, width, height, "Iy");
            compare(It, It_cpu, width, height, "It");
        }

        /*writeToFile(Ix, width, height, "Ix.csv");
        writeToFile(Iy, width, height, "Iy.csv");
        writeToFile(It, width, height, "It.csv");*/
#endif

        dim3 threadsPerBlock2(BLOCK_SIZE, BLOCK_SIZE / 4);
        dim3 numBlocks2(width / threadsPerBlock2.x, height / threadsPerBlock2.y);
        cudaComputeOpticalFlow2 << <numBlocks2, threadsPerBlock2 >> > (d_grad, width, height, stride, d_u, d_v);

        cudaMemcpy(u_cpu, d_u, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(v_cpu, d_v, size, cudaMemcpyDeviceToHost);

#if DISPLAY_STREAMS
        visualizeOpticalFlow(u_cpu, v_cpu, width, height, stride, output);
#endif 

#if DEBUG
        if (frame_num > 1)
        {
            compare(u_cpu, u, width, height, "U");
            compare(v_cpu, v, width, height, "V");
        }

        /* writeToFile(u, width, height, "U.csv");
         writeToFile(v, width, height, "V.csv");

         writeToFile(u_cpu, width, height, "U_gpu.csv");
         writeToFile(v_cpu, width, height, "V_gpu.csv");*/
#endif

        time_t tock = time(NULL);
        calc_fps = frame_num / (float)difftime(tock, tick);

#if DISPLAY_STREAMS
        cv::Mat opflow(height, width, CV_8UC3, output);
        cv::cvtColor(opflow, opflow, cv::COLOR_HSV2BGR);

        char text[12];
        sprintf(text, "FPS: %.2f", calc_fps);

        cv::putText(opflow, //target image
            text, //text
            cv::Point(10, opflow.rows - 10), //top-left position
            cv::FONT_HERSHEY_DUPLEX,
            1.0,
            CV_RGB(118, 185, 0), //font color
            1);

        cv::imshow("Optical Flow Display", opflow);
#endif
    }

    cout << "Calculated FPS: " << calc_fps << "\n";

    free(I1);
    free(I2);
    free(Ix);
    free(Iy);
    free(It);
    free(u);
    free(v);

    cudaFree(d_I1);
    cudaFree(d_I2);
    cudaFree(d_grad);
    cudaFree(d_u);
    cudaFree(d_v);

    free(grad_cpu);
    free(Ix_cpu);
    free(Iy_cpu);
    free(It_cpu);
    free(u_cpu);
    free(v_cpu);

}