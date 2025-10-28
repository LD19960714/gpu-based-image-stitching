// Created by s1nh.org.

#ifndef IMAGE_STITCHING_APP_H
#define IMAGE_STITCHING_APP_H

#include "opencv2/opencv.hpp"

#include "sensor_data_interface.h"
#include "image_stitcher.h"

class App {
public:
  App();
  App(int );
  [[noreturn]] void run_stitching();
  void run_single_thread_stitch();

private:
  SensorDataInterface sensor_data_interface_;
  ImageStitcher image_stitcher_;
  std::vector<cv::Mat> image_vector_;
  cv::UMat image_concat_umat_;
  int total_cols_;

};

#endif //IMAGE_STITCHING_APP_H
