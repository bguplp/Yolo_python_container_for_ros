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
Compiling on Linux by using command make (or alternative way by using command: cmake . && make ):
```bash
$ cd ~/
$ git clone https://github.com/bguplp/Yolo_python_container_for_ros.git
$ cd Yolo_python_container_for_ros/src/ros_yolo/src/darknet/
$ make
$ cd ../../../..
$ source /opt/ros/kinetic/setup.bash
$ rm -rv build
$ rm -rv devel
$ catkin_make
$ cd src/
$ mkdir tmp
$ tar -xvzf ros_yolo.tar.gz -C tmp ros_yolo/src/darknet/cfg ros_yolo/src/darknet/data
$ cp -a tmp/ros_yolo/src/darknet/cfg tmp/ros_yolo/src/darknet/data ros_yolo/src/darknet
$ rm -rv tmp
$ sed -i "s|$names = /home/lar0/alex_project_ros_ws/src/ros_yolo/src/darknet/data/coco.names|$names = $HOME/Yolo_python_container_for_ros/src/ros_yolo/src/darknet/data/coco.names|" ros_yolo/src/darknet/cfg/coco.data
$ sudo apt-get install ros-kinetic-vision-msgs
```

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
3) Run the alex_yolo.py script with the command rosrun alex_yolo.py.

## Training your own model:

1. Create file `yolo-obj.cfg` with the same content as in `yolov3.cfg` (or copy `yolov3.cfg` to `yolo-obj.cfg)` and:

  * change line batch to [`batch=64`]
  * change line subdivisions to [`subdivisions=8`]
  * change line max_batches to (`classes*2000` but not less than `4000`), f.e. [`max_batches=6000`]
    if you train for 3 classes
  * change line steps to 80% and 90% of max_batches, f.e. [`steps=4800,5400`]
  * change line `classes=80` to your number of objects in each of 3 `[yolo]`-layers:
  * change [`filters=255`] to filters=(classes + 5)x3 in the 3 `[convolutional]` before each `[yolo]` layer
  So if `classes=1` then should be `filters=18`. If `classes=2` then write `filters=21`.
  
  **(Do not write in the cfg-file: filters=(classes + 5)x3)**
  
  (Generally `filters` depends on the `classes`, `coords` and number of `mask`s, i.e. filters=`(classes + coords + 1)*<number of mask>`, where `mask` is indices of anchors. If `mask` is absence, then filters=`(classes + coords + 1)*num`)

  So for example, for 2 objects, your file `yolo-obj.cfg` should differ from `yolov3.cfg` in such lines in each of **3** [yolo]-layers:

  ```
  [convolutional]
  filters=21

  [region]
  classes=2
  ```

2. Create file `obj.names` in the directory `build\darknet\x64\data\`, with objects names - each in new line

3. Create file `obj.data` in the directory `build\darknet\x64\data\`, containing (where **classes = number of objects**):

  ```
  classes= 2
  train  = data/train.txt
  valid  = data/test.txt
  names = data/obj.names
  backup = backup/
  ```

4. Put image-files (.jpg) of your objects in the directory `/src/ros_yolo/src/darknet/\data\obj\`

5. You should label each object on images from your dataset. Use this visual GUI-software for marking bounded boxes of objects and generating annotation files for Yolo v2 & v3.

It will create `.txt`-file for each `.jpg`-image-file - in the same directory and with the same name, but with `.txt`-extension, and put to file: object number and object coordinates on this image, for each object in new line: 

`<object-class> <x_center> <y_center> <width> <height>`

  Where: 
  * `<object-class>` - integer object number from `0` to `(classes-1)`
  * `<x_center> <y_center> <width> <height>` - float values **relative** to width and height of image, it can be equal from `(0.0 to 1.0]`
  * for example: `<x> = <absolute_x> / <image_width>` or `<height> = <absolute_height> / <image_height>`
  * atention: `<x_center> <y_center>` - are center of rectangle (are not top-left corner)

  For example for `img1.jpg` you will be created `img1.txt` containing:

  ```
  1 0.716797 0.395833 0.216406 0.147222
  0 0.687109 0.379167 0.255469 0.158333
  1 0.420312 0.395833 0.140625 0.166667
  ```

6. Create file `train.txt` in directory `/src/ros_yolo/src/darknet/data`, with filenames of your images, each filename in new line, with path relative to `./darknet`, for example containing:

  ```
  data/obj/img1.jpg
  data/obj/img2.jpg
  data/obj/img3.jpg
  ```

7. Download pre-trained weights for the convolutional layers (154 MB): https://pjreddie.com/media/files/darknet53.conv.74 and put to the directory `build\darknet\x64`

8. Start training by using the command line: `./darknet detector train data/obj.data yolo-obj.cfg darknet53.conv.74`
     
   * (file `yolo-obj_last.weights` will be saved to the `/src/ros_yolo/src/darknet/backup` for each 100 iterations)
   * (file `yolo-obj_xxxx.weights` will be saved to the `/src/ros_yolo/src/darknet/backup` for each 1000 iterations)
   * (to disable Loss-Window use `./darknet detector train data/obj.data yolo-obj.cfg darknet53.conv.74 -dont_show`, if you train on computer without monitor like a cloud Amazon EC2)
   * (to see the mAP & Loss-chart during training on remote server without GUI, use command `./darknet detector train data/obj.data yolo-obj.cfg darknet53.conv.74 -dont_show -mjpeg_port 8090 -map` then open URL `http://ip-address:8090` in Chrome/Firefox browser)

8.1. For training with mAP (mean average precisions) calculation for each 4 Epochs (set `valid=valid.txt` or `train.txt` in `obj.data` file) and run: `./darknet detector train data/obj.data yolo-obj.cfg darknet53.conv.74 -map`

9. After training is complete - get result `yolo-obj_final.weights` from path `build\darknet\x64\backup\`

 * After each 100 iterations you can stop and later start training from this point. For example, after 2000 iterations you can stop training, and later just start training using: `./darknet detector train data/obj.data yolo-obj.cfg backup\yolo-obj_2000.weights`

    (in the original repository https://github.com/pjreddie/darknet the weights-file is saved only once every 10 000 iterations `if(iterations > 1000)`)

 * Also you can get result earlier than all 45000 iterations.
 
 **Note:** If during training you see `nan` values for `avg` (loss) field - then training goes wrong, but if `nan` is in some other lines - then training goes well.
 
 **Note:** If you changed width= or height= in your cfg-file, then new width and height must be divisible by 32.
 
 **Note:** After training use such command for detection: `./darknet detector test data/obj.data yolo-obj.cfg yolo-obj_8000.weights`
 
  **Note:** if error `Out of memory` occurs then in `.cfg`-file you should increase `subdivisions=16`, 32 or 64: [link](https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L4)
 
### How to train tiny-yolo (to detect your custom objects):

Do all the same steps as for the full yolo model as described above. With the exception of:
* Download default weights file for yolov3-tiny: https://pjreddie.com/media/files/yolov3-tiny.weights
* Get pre-trained weights `yolov3-tiny.conv.15` using command: `./darknet partial cfg/yolov3-tiny.cfg yolov3-tiny.weights yolov3-tiny.conv.15 15`
* Make your custom model `yolov3-tiny-obj.cfg` based on `cfg/yolov3-tiny_obj.cfg` instead of `yolov3.cfg`
* Start training: `./darknet detector train data/obj.data yolov3-tiny-obj.cfg yolov3-tiny.conv.15`
 
## When to stop training:

Usually sufficient 2000 iterations for each class(object), but not less than 4000 iterations in total. But for a more precise definition when you should stop training, use the following manual:

1. During training, you will see varying indicators of error, and you should stop when no longer decreases **0.XXXXXXX avg**:

  > Region Avg IOU: 0.798363, Class: 0.893232, Obj: 0.700808, No Obj: 0.004567, Avg Recall: 1.000000,  count: 8
  > Region Avg IOU: 0.800677, Class: 0.892181, Obj: 0.701590, No Obj: 0.004574, Avg Recall: 1.000000,  count: 8
  >
  > **9002**: 0.211667, **0.60730 avg**, 0.001000 rate, 3.868000 seconds, 576128 images
  > Loaded: 0.000000 seconds

  * **9002** - iteration number (number of batch)
  * **0.60730 avg** - average loss (error) - **the lower, the better**

  When you see that average loss **0.xxxxxx avg** no longer decreases at many iterations then you should stop training. The final avgerage loss can be from `0.05` (for a small model and easy dataset) to `3.0` (for a big model and a difficult dataset).

2. Once training is stopped, you should take some of last `.weights`-files from `/src/ros_yolo/src/darknet/backup` and choose the best of them:

For example, you stopped training after 9000 iterations, but the best result can give one of previous weights (7000, 8000, 9000). It can happen due to overfitting. **Overfitting** - is case when you can detect objects on images from training-dataset, but can't detect objects on any others images. You should get weights from **Early Stopping Point**:

![Overfitting](https://hsto.org/files/5dc/7ae/7fa/5dc7ae7fad9d4e3eb3a484c58bfc1ff5.png) 

To get weights from Early Stopping Point:

  2.1. At first, in your file `obj.data` you must specify the path to the validation dataset `valid = valid.txt` (format of `valid.txt` as in `train.txt`), and if you haven't validation images, just copy `data\train.txt` to `data\valid.txt`.

  2.2 If training is stopped after 9000 iterations, to validate some of previous weights use this commands:

* `./darknet detector map data/obj.data yolo-obj.cfg backup\yolo-obj_7000.weights`
* `./darknet detector map data/obj.data yolo-obj.cfg backup\yolo-obj_8000.weights`
* `./darknet detector map data/obj.data yolo-obj.cfg backup\yolo-obj_9000.weights`

And comapre last output lines for each weights (7000, 8000, 9000):

Choose weights-file **with the highest mAP (mean average precision)** or IoU (intersect over union)

For example, **bigger mAP** gives weights `yolo-obj_8000.weights` - then **use this weights for detection**.

Or just train with `-map` flag: 

`./darknet detector train data/obj.data yolo-obj.cfg darknet53.conv.74 -map` 

So you will see mAP-chart (red-line) in the Loss-chart Window. mAP will be calculated for each 4 Epochs using `valid=valid.txt` file that is specified in `obj.data` file (`1 Epoch = images_in_train_txt / batch` iterations)

![loss_chart_map_chart](https://hsto.org/webt/yd/vl/ag/ydvlagutof2zcnjodstgroen8ac.jpeg)

Example of custom object detection: `./darknet detector test data/obj.data yolo-obj.cfg yolo-obj_8000.weights`

* **IoU** (intersect over union) - average instersect over union of objects and detections for a certain threshold = 0.24

* **mAP** (mean average precision) - mean value of `average precisions` for each class, where `average precision` is average value of 11 points on PR-curve for each possible threshold (each probability of detection) for the same class (Precision-Recall in terms of PascalVOC, where Precision=TP/(TP+FP) and Recall=TP/(TP+FN) ), page-11: http://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc09.pdf

**mAP** is default metric of precision in the PascalVOC competition, **this is the same as AP50** metric in the MS COCO competition.
In terms of Wiki, indicators Precision and Recall have a slightly different meaning than in the PascalVOC competition, but **IoU always has the same meaning**.

![precision_recall_iou](https://hsto.org/files/ca8/866/d76/ca8866d76fb840228940dbf442a7f06a.jpg)

## How to improve object detection:

### Before training:
  * set flag `random=1` in your `.cfg`-file - it will increase precision by training Yolo for different resolutions: 

  * increase network resolution in your `.cfg`-file (`height=608`, `width=608` or any value multiple of 32) - it will increase precision

  * my Loss is very high and mAP is very low, is training wrong? Run training with ` -show_imgs` flag at the end of training command, do you see correct bounded boxes of objects (in windows or in files `aug_...jpg`)? If no - your training dataset is wrong.

  * for each object which you want to detect - there must be at least 1 similar object in the Training dataset with about the same: shape, side of object, relative size, angle of rotation, tilt, illumination. So desirable that your training dataset include images with objects at diffrent: scales, rotations, lightings, from different sides, on different backgrounds - you should preferably have 2000 different images for each class or more, and you should train `2000*classes` iterations or more

  * desirable that your training dataset include images with non-labeled objects that you do not want to detect - negative samples without bounded box (empty `.txt` files) - use as many images of negative samples as there are images with objects

  * What is the best way to mark objects: label only the visible part of the object, or label the visible and overlapped part of the object, or label a little more than the entire object (with a little gap)? Mark as you like - how would you like it to be detected.

  * for training with a large number of objects in each image, add the parameter `max=200` or higher value in the last `[yolo]`-layer or `[region]`-layer in your cfg-file (the global maximum number of objects that can be detected by YoloV3 is `0,0615234375*(width*height)` where are width and height are parameters from `[net]` section in cfg-file) 
  
  * General rule - your training dataset should include such a set of relative sizes of objects that you want to detect: 

    * `train_network_width * train_obj_width / train_image_width ~= detection_network_width * detection_obj_width / detection_image_width`
    * `train_network_height * train_obj_height / train_image_height ~= detection_network_height * detection_obj_height / detection_image_height`
    
    I.e. for each object from Test dataset there must be at least 1 object in the Training dataset with the same class_id and about the same relative size:

    `object width in percent from Training dataset` ~= `object width in percent from Test dataset` 
   
    That is, if only objects that occupied 80-90% of the image were present in the training set, then the trained network will not be able to detect objects that occupy 1-10% of the image.
    
  * to speedup training (with decreasing detection accuracy) do Fine-Tuning instead of Transfer-Learning, set param `stopbackward=1`,
    then do this command: `./darknet partial cfg/yolov3.cfg yolov3.weights yolov3.conv.81 81` will be created file `yolov3.conv.81`,
    then train by using weights file `yolov3.conv.81` instead of `darknet53.conv.74`

  * each: `model of object, side, illimination, scale, each 30 grad` of the turn and inclination angles - these are *different objects* from an internal perspective of the neural network. So the more *different objects* you want to detect, the more complex network model should be used.

  * Only if you are an **expert** in neural detection networks - recalculate anchors for your dataset for `width` and `height` from cfg-file:
  `./darknet detector calc_anchors data/obj.data -num_of_clusters 9 -width 416 -height 416`
   then set the same 9 `anchors` in each of 3 `[yolo]`-layers in your cfg-file. But you should change indexes of anchors `masks=` for each [yolo]-layer, so that 1st-[yolo]-layer has anchors larger than 60x60, 2nd larger than 30x30, 3rd remaining. Also you should change the `filters=(classes + 5)*<number of mask>` before each [yolo]-layer. If many of the calculated anchors do not fit under the appropriate layers - then just try using all the default anchors.


### After training - for detection:

  * Increase network-resolution by set in your `.cfg`-file (`height=608` and `width=608`) or (`height=832` and `width=832`) or (any value multiple of 32) - this increases the precision and makes it possible to detect small objects.
  
    * it is not necessary to train the network again, just use `.weights`-file already trained for 416x416 resolution
    * but to get even greater accuracy you should train with higher resolution 608x608 or 832x832, note: if error `Out of memory` occurs then in `.cfg`-file you should increase `subdivisions=16`, 32 or 64.
 
