Report - <a href="CUDA Optical Flow.pdf">Link</a>
Video Submission - <a href="https://www.youtube.com/watch?v=wG-psnHYOsk">https://www.youtube.com/watch?v=wG-psnHYOsk</a>

<a href="https://www.youtube.com/watch?v=wG-psnHYOsk" target="_blank"><img src="readme_img/first.png"></a>

<a href="https://youtu.be/huYqKNw96ZA?feature=shared" target="_blank"><img src="readme_img/second.png"></a>

# Create a New CUDA Runtime Project (ignore if using the .sln file)

File -> New -> Project -> CUDA Runtime
<img src="readme_img/img2.png">

# OpenCV Installation

Our project utilizes OpenCV. First step to perform is to install OpenCV from <a href="https://opencv.org/releases/">[OpenCV Releases](https://opencv.org/releases/)</a> and a add -

`<YOUR OPENCV FOLDER>\opencv\build\x64\vc16\bin`

to the `Path` (of your environment variables)

# Adding OpenCV to Visual Studio Project

1. VC++ Directories -> Include Directories -> Add path to `include` folder in opencv.
   
<img src="readme_img/image1.png">

2. VC++ Directories -> Include Directories -> Add path to `opencv/build/x64/vc16/lib` folder in opencv.

<img src="readme_img/image2.png">

3. Linker -> Input -> Additional Dependencies, add the following paths and libraries -

<img src="readme_img/image3.png">

<b>At this point, OpenCV should be added to the project.</b>

# Setup Debugging

The main function expects a video stream as one of the arguments. For the purposes of debugging, it can be added in 

Debugging -> Commad Arguments

<img src="readme_img/img1.png">