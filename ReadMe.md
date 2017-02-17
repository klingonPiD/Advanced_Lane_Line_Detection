# Project 3: Advanced Lane Finding
### Ajay Paidi

# Objective
The objective of this project is to develop a software pipeline to identify lane boundaries in a video from a front-facing camera on a car. The software pipeline will use traditional (but advanced) computer vision techniques to detect lane lines and should be reasonably robust enough to handle different situations (curves, lighting conditions, etc.)

# File structure
- **ReadMe.md**: This file
- **calibrate_images**: Script that performs camera calibration and saves the camera matrix and distortion params.
- **preprocess_lane_lines.py**: Script that performs pre-processing (edge detection, color thresholding, etc.) on the images.
- **ProcessFrame.py**: Class with methods for detecting lane lines and computing lane statistics.
- **visualize_images.py**: Utility script for visualizing images and lane boundaries.
- **advanced_lane_detect_pipeline.py**: Main script that implements the software pipeline to detect lane lines on videos.
- **demo_advanced_lane_lines.ipynb**: Python notebook that demonstrates some of the concepts in the software pipeline.

# Approach

My approach to solving the problem can be broadly categorized into these two categories
### Camera calibration
This corrects for distortion introduced in images by the camera lens and computes a transformation between image co-ordinates and real world co-ordinates.

### Implementing a software pipeline
 The software pipeline essentially does two main sets of activities on the incoming stream of images
 1. **Pre-processing**:  This involves undistorting the image, performing edge detection, thresholding desired colors in different color spaces, and finally applying a perspective transform to get a 'birds-eye-view' of the binary image.
 2. **Processing**: This involves taking a histogram, applying a sliding window technique, and calculating polynomial co-efficients to detect and draw the lane lines. In addition a moving average scheme is implemented to smooth out the jitter in the video frames.

 All the above steps are illustrated with sutiable examples in the `demo_advanced_lane_lines.ipynb` notebook.

# Results

[![Project Video](https://img.youtube.com/vi/fF09efg_VTk/0.jpg)](https://youtu.be/fF09efg_VTk)

[![Challenge Video](https://img.youtube.com/vi/ogEaJFM0RbM/0.jpg)](https://youtu.be/ogEaJFM0RbM)

# Discussion

 This was a tedious project that involved piecing together several tiny solutions to solve the problem. While the end solution performed quite well on the project video, it really struggled on the challenge videos. Any number of factors that violate the assumptions made in the individual solutions could break the pipeline. Some of these factors include unclear lane markings, sharp turns / hair pin bends, excessive brightness or darkness, shadows, objects obstructing the camera's field of view (like another car), etc. Some possible ways to make the pipeline more robust include
 1. Using a dynamic thresholding scheme - Automatically choose different thresholds for edge detection and color extraction based on the type of the input image (dark, bright, different colo lane lines, etc.).
 2. Outlier removal - Use the computed lane statistics to throw away results that deviate a lot from the immediately prior results.
 3. Occlusion removal - Detect and remove objects that obstruct the camera's field of view.


 A potential (and exciting!) solution would be to tackle this problem using deep learning approaches. This might remove the need to devise several individual solutions. However it would require that one has enough data to represent all possible road conditions.

 # References

Most of the code in the Udacity lecture notes was used as a starting material for this project.
