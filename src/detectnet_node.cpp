/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2016 ThundeRatz

 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <sensor_msgs/RegionOfInterest.h>

#include <string>
#include <vector>

#include "ros_detectnet/detector.h"

namespace {
  Detector* detector;
  ros::Publisher publisher;

  void publishRegions(const std::vector<sensor_msgs::RegionOfInterest>& predictions)  {
    for (auto& roi : predictions)
      publisher.publish(roi);
  }

  void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
    cv_bridge::CvImageConstPtr imagePtr;
    try {
      imagePtr = cv_bridge::toCvShare(msg, "bgr8");
    } catch (cv_bridge::Exception& e) {
      ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
    const cv::Mat& image = imagePtr->image;
    publishRegions(detector->detect(image));
  }
}  // namespace

int main(int argc, char **argv) {
  ros::init(argc, argv, "detectnet_node");

  const std::string NET_DATA = ros::package::getPath("detectnet") + "/data/";
  detector = new Detector(NET_DATA + "deploy.prototxt", NET_DATA + "/snapshot.caffemodel");

  ros::NodeHandle node;
  image_transport::ImageTransport transport(node);
  transport.subscribe("image", 1, imageCallback);
  publisher = node.advertise<sensor_msgs::RegionOfInterest>("detectnet", 100);
  ros::spin();

  delete detector;
  return 0;
}
