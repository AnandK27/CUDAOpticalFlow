#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

using namespace std;

#define WIN_SIZE 5  // Window size for Lucas-Kanade method
#define BLOCK_SIZE 32

#define DISPLAY_STREAMS true

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

    double fps = cap.get(cv::CAP_PROP_FPS);
    int delay = static_cast<int>(1000 / fps); // Delay between frames in milliseconds

    
    // Create a video writer object
    cv::VideoWriter writer("output.mp4", cv::VideoWriter::fourcc('h', '2', '6', '4'), fps, cv::Size(width * 2, height));

    if (!writer.isOpened()) {
        std::cerr << "Error opening video writer" << std::endl;
        return -1;
    }


    // CPU Memory Allocation
    cv::Mat frame;
    cv::Mat res;
    unsigned char* temp = NULL;
    unsigned char* I1 = (unsigned char*)malloc(height * stride);
    unsigned char* I2 = (unsigned char*)malloc(height * stride);
    float* Ix = (float*)calloc(height * stride, sizeof(float));
    float* Iy = (float*)calloc(height * stride, sizeof(float));
    float* It = (float*)calloc(height * stride, sizeof(float));
    float* u = (float*)calloc(height * stride, sizeof(float));
    float* v = (float*)calloc(height * stride, sizeof(float));
    unsigned char* output = (unsigned char*)calloc(height * stride * 3, sizeof(unsigned char));

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
        

        if (cv::waitKey(delay) == 'q') {
            std::cout << "Exiting video playback" << std::endl;
            break;
        }
#endif
        frame_num++;
        temp = I1;
        I1 = I2;
        I2 = temp;
        memcpy(I2, frame.data, height * width * sizeof(unsigned char));

        computeGradients(I2, width, height, stride, Ix, Iy, It, I1);
        computeOpticalFlow(Ix, Iy, It, width, height, stride, u, v);
#if DISPLAY_STREAMS
        visualizeOpticalFlow(u, v, width, height, stride, output);
#endif

        time_t tock = time(NULL);
        calc_fps = frame_num / (float)difftime(tock, tick);

#if DISPLAY_STREAMS
        cv::Mat opflow(height, width, CV_8UC3, output);
        cv::cvtColor(opflow, opflow, cv::COLOR_HSV2BGR);

        cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
        std::vector<cv::Mat> mats = {frame, opflow};
        cv::hconcat(mats, res);
        writer.write(res);

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

    cap.release();
    writer.release();
    
    cout << "Calculated FPS: " << calc_fps << "\n";

    free(I1);
    free(I2);
    free(Ix);
    free(Iy);
    free(It);
    free(u);
    free(v);
}