## Work in progress - The message format will change in the future, do not rely on this package!

# ROS DetectNet integration
Integrates DetectNet models (eg. generated with [DIGITS](https://github.com/NVIDIA/DIGITS/)) with ROS, generating messages with position and confidence of detection.

## Requirements
* ROS (tested on kinetic)

* Caffe with GPU support

* CUDA

## Installation
This repository should be cloned to the `src` directory of your workspace. Alternatively, the following `.rosinstall` entry adds this repository for usage with `wstool`:

```
- git:
    local-name: detectnet
    uri: https://github.com/ThundeRatz/ros_detectnet.git
```

## Setup
Copy your model `deploy.prototxt` and a snapshot of the network's weights as `snapshot.caffemodel` to the data folder.

## ROS Topics
Raw images are received in the `image` topic and results are published as [`sensor_msgs/RegionOfInterest`](http://docs.ros.org/kinetic/api/sensor_msgs/html/msg/RegionOfInterest.html) messages at `detectnet`. This will change in the future to a custom message with more information, such as confidence.

For example, the following launch file can be used to map a images from [usb_cam](http://wiki.ros.org/usb_cam) to detectnet:

```
<launch>
  <group ns="vision">
    <remap from="image" to="usb_cam/image_raw" />
    <node pkg="usb_cam" name="usb_cam" type="usb_cam_node" required="true" />
    <node pkg="detectnet" name="detectnet" type="detectnet_node" required="true" />
  </group>
</launch>

```

## Contributors
Written by the ThundeRatz robotics team.

Caffe code is based on [ros_caffe](https://github.com/tzutalin/ros_caffe), which provides ROS integration with Caffe for image classification.
