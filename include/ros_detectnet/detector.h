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

#ifndef ROS_DETECTNET_DETECTOR_H
#define ROS_DETECTNET_DETECTOR_H

#include <vector>
#include <string>
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <sensor_msgs/RegionOfInterest.h>

namespace detectnet
{
class Detector
{
 public:
  Detector(const std::string& model_file, const std::string& trained_file);
  std::vector<sensor_msgs::RegionOfInterest> detect(const cv::Mat& img);

 private:
  std::vector<float> forward(const cv::Mat& img);
  void wrapInputLayer(std::vector<cv::Mat>* input_channels);
  void preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);

 private:
  caffe::shared_ptr<caffe::Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
};
}  // namespace detectnet

#endif  // ROS_DETECTNET_DETECTOR_H
