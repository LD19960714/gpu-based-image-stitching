// Created by s1nh.org.

#include "app.h"

#include <iostream>
#include <mutex>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>

#include "image_stitcher.h"
#include "stitching_param_generater.h"

App::App(int ) {


}

App::App() {
  sensor_data_interface_.InitVideoCapture();

  std::vector<cv::UMat> first_image_vector = std::vector<cv::UMat>(sensor_data_interface_.num_img_);
  std::vector<cv::Mat> first_mat_vector = std::vector<cv::Mat>(sensor_data_interface_.num_img_);
  std::vector<cv::UMat> reproj_xmap_vector;
  std::vector<cv::UMat> reproj_ymap_vector;
  std::vector<cv::UMat> undist_xmap_vector;
  std::vector<cv::UMat> undist_ymap_vector;
  std::vector<cv::Rect> image_roi_vect;

  std::vector<std::mutex> image_mutex_vector(sensor_data_interface_.num_img_);
  sensor_data_interface_.get_image_vector(first_image_vector, image_mutex_vector);

  for (size_t i = 0; i < sensor_data_interface_.num_img_; ++i) {
    first_image_vector[i].copyTo(first_mat_vector[i]);
  }

  StitchingParamGenerator stitching_param_generator(first_mat_vector);

  stitching_param_generator.GetReprojParams(
      undist_xmap_vector,
      undist_ymap_vector,
      reproj_xmap_vector,
      reproj_ymap_vector,
      image_roi_vect
  );

  image_stitcher_.SetParams(
      100,
      undist_xmap_vector,
      undist_ymap_vector,
      reproj_xmap_vector,
      reproj_ymap_vector,
      image_roi_vect
  );
  total_cols_ = 0;
  for (size_t i = 0; i < sensor_data_interface_.num_img_; ++i) {
    total_cols_ += image_roi_vect[i].width;
  }
  image_concat_umat_ = cv::UMat(image_roi_vect[0].height, total_cols_, CV_8UC3);
}

[[noreturn]] void App::run_stitching() {
  std::vector<cv::UMat> image_vector(sensor_data_interface_.num_img_);
  std::vector<std::mutex> image_mutex_vector(sensor_data_interface_.num_img_);
  std::vector<cv::UMat> images_warped_vector(sensor_data_interface_.num_img_);
  std::thread record_videos_thread(
      &SensorDataInterface::RecordVideos,
      &sensor_data_interface_
  );

  int64_t t0, t1, t2, t3, tn;

  size_t frame_idx = 0;
  while (true) {
    t0 = cv::getTickCount();

    std::vector<std::thread> warp_thread_vect;
    sensor_data_interface_.get_image_vector(image_vector, image_mutex_vector);
    t1 = cv::getTickCount();

    for (size_t img_idx = 0; img_idx < sensor_data_interface_.num_img_; ++img_idx) {
      warp_thread_vect.emplace_back(       
          &ImageStitcher::WarpImages,
          &image_stitcher_,
          img_idx,
          20,
          image_vector,
          std::ref(image_mutex_vector),
          std::ref(images_warped_vector),
          std::ref(image_concat_umat_)
      );
    }
    for (auto& warp_thread : warp_thread_vect) {
      warp_thread.join();
    }
    t2 = cv::getTickCount();

    imwrite("../results/image_concat_umat_" + std::to_string(frame_idx) + ".png",
            image_concat_umat_);

    frame_idx++;
    tn = cv::getTickCount();

    std::cout << "[app] "
              << double(t1 - t0) / cv::getTickFrequency() << ";"
              << double(t2 - t1) / cv::getTickFrequency() << ";"
              << 1 / (double(t2 - t0) / cv::getTickFrequency()) << " FPS; "
              << 1 / (double(tn - t0) / cv::getTickFrequency()) << " Real FPS." << std::endl;

  }
//  record_videos_thread.join();
}
void App::run_single_thread_stitch()
{
  
  cv::VideoWriter writer(
        "../res/stitched_video.avi",
        cv::VideoWriter::fourcc('M','J','P','G'), // 编码器
        30, // 帧率，可根据实际视频改
        cv::Size(1920, 540)
    );
  

  std::string video_dir = "/home/ld/data/视频拼接/";
  std::vector<std::string> video_file_name = {"left.ts", "right.mp4"};
  // std::string video_dir = "/home/ld/project/gpu-based-image-stitching/datasets/air-4cam-mp4/";
  // std::vector<std::string> video_file_name = {"00.mp4", "01.mp4"};
  std::vector<cv::VideoCapture> video_capture_vectors;
  std::vector<cv::Mat> imgs(video_file_name.size());
    // Init video capture.
  for (int i = 0; i < video_file_name.size(); ++i) {

      std::string file_name = video_dir + video_file_name[i];
      cv::VideoCapture capture(file_name);
      cv::Mat frame;
      if (!capture.isOpened())
        std::cout << "Failed to open capture " << i << std::endl;
      video_capture_vectors.emplace_back(capture);
      capture.read(frame);
      //cv::resize(frame, frame, cv::Size(960,540));
      imgs[i] = frame.clone();

  }


  std::vector<cv::UMat> reproj_xmap_vector;
  std::vector<cv::UMat> reproj_ymap_vector;
  std::vector<cv::UMat> undist_xmap_vector;
  std::vector<cv::UMat> undist_ymap_vector;
  std::vector<cv::Rect> image_roi_vect;

  // 初始化：第一次特征匹配和参数估计
  StitchingParamGenerator stitching_param_generator(imgs);

  stitching_param_generator.GetReprojParams(
      undist_xmap_vector,
      undist_ymap_vector,
      reproj_xmap_vector,
      reproj_ymap_vector,
      image_roi_vect
  );

  image_stitcher_.SetParams(
      100,
      undist_xmap_vector,
      undist_ymap_vector,
      reproj_xmap_vector,
      reproj_ymap_vector,
      image_roi_vect
  );
  total_cols_ = 0;
  for (size_t i = 0; i < video_file_name.size(); ++i) {
    total_cols_ += image_roi_vect[i].width;
  }
  image_concat_umat_ = cv::UMat(image_roi_vect[0].height, total_cols_, CV_8UC3);


  bool flag = true;
  std::vector<cv::UMat> umats(video_file_name.size());
  int frame_count = 0;  // 帧计数器
  const int RECALIBRATE_INTERVAL = 60;  // 增大间隔减少抖动
  
  // 用于平滑参数更新
  std::vector<cv::UMat> prev_reproj_xmap_vector;
  std::vector<cv::UMat> prev_reproj_ymap_vector;
  bool has_prev_params = false;
  const float SMOOTH_ALPHA = 0.6f;  // 平滑系数：0.3表示30%新参数+70%旧参数
  
  std::cout << "\n[INFO] 智能重标定已启用，每 " << RECALIBRATE_INTERVAL << " 帧更新一次参数" << std::endl;
  std::cout << "[INFO] 参数平滑系数: " << SMOOTH_ALPHA << " (减少抖动)" << std::endl;
  std::cout << "[INFO] 开始视频拼接...\n" << std::endl;
  
  while (true) {

    for (int i = 0; i < video_file_name.size(); ++i) {
      
        cv::UMat frame;
        if(!video_capture_vectors[i].read(frame)) {

            flag = false;
            break;
        }else {
          //cv::resize(frame, frame, cv::Size(960,540));
          umats[i] = frame;

        }
    }
    if(!flag)
          break;
    
    // 每90帧重新进行特征匹配和参数估计
    if (frame_count % RECALIBRATE_INTERVAL == 0 && frame_count > 0) {  // 跳过第0帧
      int64_t recalib_start = cv::getTickCount();
      
      std::cout << "\n[RECALIBRATE] Frame " << frame_count << ": 重新进行特征匹配..." << std::endl;
      
      // 将当前帧转换为Mat以便进行特征检测
      std::vector<cv::Mat> current_frames(video_file_name.size());
      for (int i = 0; i < video_file_name.size(); ++i) {
        umats[i].copyTo(current_frames[i]);
      }
      
      // 保存旧参数用于平滑
      if (has_prev_params) {
        prev_reproj_xmap_vector = reproj_xmap_vector;
        prev_reproj_ymap_vector = reproj_ymap_vector;
      }
      
      // 重新创建StitchingParamGenerator并计算新参数
      StitchingParamGenerator new_generator(current_frames);
      
      std::vector<cv::UMat> new_reproj_xmap_vector;
      std::vector<cv::UMat> new_reproj_ymap_vector;
      
      new_generator.GetReprojParams(
          undist_xmap_vector,
          undist_ymap_vector,
          new_reproj_xmap_vector,
          new_reproj_ymap_vector,
          image_roi_vect
      );
      
      // 关键优化：参数平滑过渡，减少抖动
      if (has_prev_params) {
        // 检查尺寸是否匹配
        bool size_match = true;
        for (size_t i = 0; i < new_reproj_xmap_vector.size(); ++i) {
          if (new_reproj_xmap_vector[i].size() != prev_reproj_xmap_vector[i].size() ||
              new_reproj_ymap_vector[i].size() != prev_reproj_ymap_vector[i].size()) {
            size_match = false;
            std::cout << "[RECALIBRATE] 检测到ROI尺寸变化 (" << i << "): "
                      << "new=" << new_reproj_xmap_vector[i].size() << " vs prev=" 
                      << prev_reproj_xmap_vector[i].size() << "，跳过参数平滑" << std::endl;
            break;
          }
        }
        
        if (size_match) {
          std::cout << "[RECALIBRATE] 应用参数平滑..." << std::endl;
          // 先确保reproj_xmap_vector和reproj_ymap_vector有足够的空间
          reproj_xmap_vector.resize(new_reproj_xmap_vector.size());
          reproj_ymap_vector.resize(new_reproj_ymap_vector.size());
          
          for (size_t i = 0; i < new_reproj_xmap_vector.size(); ++i) {
            cv::UMat smoothed_xmap, smoothed_ymap;
            cv::addWeighted(new_reproj_xmap_vector[i], SMOOTH_ALPHA, 
                           prev_reproj_xmap_vector[i], 1.0f - SMOOTH_ALPHA, 
                           0, smoothed_xmap);
            cv::addWeighted(new_reproj_ymap_vector[i], SMOOTH_ALPHA, 
                           prev_reproj_ymap_vector[i], 1.0f - SMOOTH_ALPHA, 
                           0, smoothed_ymap);
            reproj_xmap_vector[i] = smoothed_xmap;
            reproj_ymap_vector[i] = smoothed_ymap;
          }
        } else {
          // 尺寸不匹配，直接使用新参数
          reproj_xmap_vector = new_reproj_xmap_vector;
          reproj_ymap_vector = new_reproj_ymap_vector;
        }
      } else {
        reproj_xmap_vector = new_reproj_xmap_vector;
        reproj_ymap_vector = new_reproj_ymap_vector;
        has_prev_params = true;
      }
      
      // 更新ImageStitcher的参数
      image_stitcher_.SetParams(
          200,
          undist_xmap_vector,
          undist_ymap_vector,
          reproj_xmap_vector,
          reproj_ymap_vector,
          image_roi_vect
      );
      
      // 重新计算输出缓冲区大小
      total_cols_ = 0;
      for (size_t i = 0; i < video_file_name.size(); ++i) {
        total_cols_ += image_roi_vect[i].width;
      }
      // 重新创建输出缓冲区以匹配新的ROI尺寸
      image_concat_umat_ = cv::UMat(image_roi_vect[0].height, total_cols_, CV_8UC3);
      
      int64_t recalib_end = cv::getTickCount();
      double recalib_time = (recalib_end - recalib_start) / cv::getTickFrequency();
      std::cout << "[RECALIBRATE] 参数更新完成! 新输出尺寸: " << total_cols_ << "x" << image_roi_vect[0].height 
                << ", 耗时: " << recalib_time << " 秒\n" << std::endl;
    }
    
    for(int i = 0; i < video_file_name.size(); ++i ) {

      image_stitcher_.WarpImagesv2(i, umats, image_concat_umat_);

    }
    cv::Mat res;
    cv::resize(image_concat_umat_, res, cv::Size(1920, 540));
    cv::imwrite("../res/res.jpg", res);
    //exit(0);
    writer.write(res);
    
    frame_count++;  // 帧计数器递增

  }
    
  

}



int main() {
  //App app;
  //app.run_stitching();
  App app(0);
  app.run_single_thread_stitch();
}
