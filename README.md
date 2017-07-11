# Online Face Recognition 
For PerceptIn Robotics Vision System v0.4.0
## Introduction
This face recognition project was built for **PerceptIn Visual Inertial Module SDK v0.4.0** using OpenCv built-in `face` module.
## File List
1. `build/` Project Build Directory.

2. `model/` Directory that stores all useful pre-trained models, and user trained models.

3. `raw_data/`  Directory that stors all training data (not processed)
4. `data/` Directory that stores all training data (processed)

4. `src/data_process.py` C++ file for Data Preprocessing: read in all image files and output all processed data and a .csv file that labels all images.

5. `src/simple_model_tester.cpp` C++ file for testing model functionality and model accuracy.

6. `src/train.cpp` C++ file for training face recognition model from processed data

7. `CMakeLists.txt` text cmake file.

## Requirement
The project was built and tested under Ubuntu 16.04 with following package installed. It is highly recommended to build the project under linux system.
```
OpenCV 3.2.0
libgtk-3-dev
libusb-1.0-0-dev
libblas
liblapack
PIRVS_SDK_0.4.0
```
You can download PIRVS_SDK_0.4.0 [here](https://www.perceptin.io/download)
## How to build the project
```
mkdir build
cd build
cmake ..
```
## A Simple Demo [video](https://youtu.be/ZigdNNZy6Bk)
