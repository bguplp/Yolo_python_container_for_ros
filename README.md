# Yolo_python_container_for_ros

## **Requirements and istallation:**
* Linux
* python2
* CMake >= 3.8 for modern CUDA support: https://cmake.org/download/
* CUDA https://developer.nvidia.com/cuda-toolkit-archive
* ROS >= kinetic : http://wiki.ros.org/kinetic/Installation/Ubuntu
* OpenCV preinstalled with ROS
* Cudnn >= 7 (Optional) https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html
* GPU with CC >= 3.0 https://en.wikipedia.org/wiki/CUDA#GPUs_supported
* GCC

## **How to run**
##### Compilation
Compiling on Linux by using command make (or alternative way by using command: cmake . && make ).

##### Advanced options(set in the makefile).
GPU=1
to build with CUDA to accelerate by using GPU (CUDA should be in /user/local/cuda).

CUDNN=1
to build with cuDNN to accelerate training by using GPU (cuDNN should be in /usr/local/cudnn).

OPENCV=1 
to build with OpenCV.

OPENMP=1 
to build with OpenMP support to accelerate Yolo by using multi-core CPU.

## Using a pre trained model with ROS

1) Run the a roscore command in the terminal.
2) Run the publish_image.py or your own compressed image publisher with the command rosrun publish_image.py.
3) Run the 


 
