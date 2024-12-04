#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
using namespace std;

// TODO: Make this an argument??
#define WIN_SIZE 5  // Window size for Lucas-Kanade method

// Compute image gradients using Sobel operator
void computeGradients(const unsigned char *gray, int width, int height, int stride, float *Ix, float *Iy, float *It, const unsigned char *prevGray) {
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int idx = y * stride + x;

            Ix[idx] = (gray[idx + 1] - gray[idx - 1]) / 2.0f;
            Iy[idx] = (gray[idx + stride] - gray[idx - stride]) / 2.0f;
            It[idx] = (float)(gray[idx] - prevGray[idx]);
        }
    }
}

// Lucas-Kanade method
void computeOpticalFlow(const float *Ix, const float *Iy, const float *It, int width, int height, int stride, float *u, float *v) {
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
            } else {
                u[y * stride + x] = 0;
                v[y * stride + x] = 0;
            }
        }
    }
}

// Visualize optical flow as HSV
void visualizeOpticalFlow(const float *u, const float *v, int width, int height, int stride, unsigned char *output) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * stride + x;

            float magnitude = sqrt(u[idx] * u[idx] + v[idx] * v[idx]);
            float angle = (float)atan2(v[idx], u[idx]) * 180.0f / CV_PI + 180.0f; // Convert to degrees
            //if (angle < 0) angle += 360;

            float normMagnitude = fmin(magnitude / 10.0f, 1.0f); // Clipping magnitude

            // Convert HSV to RGB
            float h = angle / 2.0f; // [0, 6)
            float s = 0.5f;
            float v = normMagnitude;

            //int hi = (int)h % 6;
            //float f = h - hi;
            //float p = v * (1 - s);
            //float q = v * (1 - f * s);
            //float t = v * (1 - (1 - f) * s);

            //float r, g, b;
            //if (hi == 0) { r = v; g = t; b = p; }
            //else if (hi == 1) { r = q; g = v; b = p; }
            //else if (hi == 2) { r = p; g = v; b = t; }
            //else if (hi == 3) { r = p; g = q; b = v; }
            //else if (hi == 4) { r = t; g = p; b = v; }
            //else { r = v; g = p; b = q; }

            output[idx * 3 + 0] = (unsigned char)(h);
            output[idx * 3 + 1] = (unsigned char)(s * 255);
            output[idx * 3 + 2] = (unsigned char)(v * 255);
        }
    }
}

int main(int argc, char **argv) {
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
    cout << width << " " << height << endl;
    int stride = width;


    cv::Mat frame;
    unsigned char *temp = NULL;
    unsigned char *prevGray = (unsigned char *)malloc(height * stride);
    unsigned char *currGray = (unsigned char *)malloc(height * stride);
    float *Ix = (float *)calloc(height * stride, sizeof(float));
    float *Iy = (float *)calloc(height * stride, sizeof(float));
    float *It = (float *)calloc(height * stride, sizeof(float));
    float *u = (float *)calloc(height * stride, sizeof(float));
    float *v = (float *)calloc(height * stride, sizeof(float));
    unsigned char *output = (unsigned char*)calloc(height * stride * 3, sizeof(unsigned char));

    while (true)
    {
        int ret = cap.read(frame);
        if (!ret) break;
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
		// resize the frame to 512x512
		cv::resize(frame, frame, cv::Size(width, height));
        cv::imshow("Frame Dispaly", frame);
        double fps = cap.get(cv::CAP_PROP_FPS);
        int delay = static_cast<int>(1000 / fps); // Delay between frames in milliseconds

        if (cv::waitKey(delay) == 'q') {
            std::cout << "Exiting video playback" << std::endl;
            break;
        }

        temp = prevGray;
        prevGray = currGray;
        currGray = temp;

        memcpy(currGray, frame.data, height * width * sizeof(unsigned char));

        computeGradients(currGray, width, height, stride, Ix, Iy, It, prevGray);
        computeOpticalFlow(Ix, Iy, It, width, height, stride, u, v);
        visualizeOpticalFlow(u, v, width, height, stride, output);

        cv::Mat opflow(height, width, CV_8UC3, output);
        cv::cvtColor(opflow, opflow, cv::COLOR_HSV2BGR);
        
        cv::imshow("Optical Flow Display", opflow);
    }
}