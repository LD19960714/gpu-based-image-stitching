// Created by s1nh.org on 2020/11/13.
// Modified from samples/cpp/stitching_detailed.cpp

#include "stitching_param_generater.h"

#include <iostream>
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <opencv2/xfeatures2d.hpp>

#include "super_glue.h"
#include "super_point.h"

#define ENABLE_LOG 0
#define LOG(msg) std::cout << msg
#define LOGLN(msg) std::cout << msg << std::endl
#define debug

using namespace std;
using namespace cv;
using namespace cv::detail;


cv::detail::MatchesInfo makeMatchesInfo(
    int idx1, int idx2,
    const cv::detail::ImageFeatures& f1,
    const cv::detail::ImageFeatures& f2,
    const std::vector<cv::DMatch>& good_matches)
{
    cv::detail::MatchesInfo mi;
    mi.src_img_idx = idx1;
    mi.dst_img_idx = idx2;
    mi.matches = good_matches;
    mi.num_inliers = (int)good_matches.size();

    if (good_matches.size() >= 4)
    {
        std::vector<cv::Point2f> pts1, pts2;
        for (auto& m : good_matches)
        {
            pts1.push_back(f1.keypoints[m.queryIdx].pt);
            pts2.push_back(f2.keypoints[m.trainIdx].pt);
        }

        cv::Mat inlier_mask;
        mi.H = cv::findHomography(pts1, pts2, cv::RANSAC, 10.0, inlier_mask);
        mi.num_inliers = cv::countNonZero(inlier_mask);
        mi.inliers_mask.assign(inlier_mask.begin<uchar>(), inlier_mask.end<uchar>());
        //mi.confidence = (double)mi.num_inliers / (double)pts1.size();
        mi.confidence = std::min(1.0, (double)mi.num_inliers  / 50.0);

    }
    else
    {
        mi.H = cv::Mat::eye(3, 3, CV_64F);
        mi.confidence = 0;
    }

    return mi;
}


StitchingParamGenerator::StitchingParamGenerator(
    const std::vector<cv::Mat>& image_vector) {

  std::cout << "[StitchingParamGenerator] Initializing..." << std::endl;

  num_img_ = image_vector.size();

  image_vector_ = image_vector;
  mask_vector_ = std::vector<cv::UMat>(num_img_);
  mask_warped_vector_ = std::vector<cv::UMat>(num_img_);
  image_size_vector_ = std::vector<cv::Size>(num_img_);
  image_warped_size_vector_ = std::vector<cv::Size>(num_img_);
  reproj_xmap_vector_ = std::vector<cv::UMat>(num_img_);
  reproj_ymap_vector_ = std::vector<cv::UMat>(num_img_);
  camera_params_vector_ =
      std::vector<cv::detail::CameraParams>(camera_params_vector_);

  projected_image_roi_refined_vect_ = std::vector<cv::Rect>(num_img_);

  for (size_t img_idx = 0; img_idx < num_img_; img_idx++) {
    image_size_vector_[img_idx] = image_vector_[img_idx].size();
  }

  std::vector<cv::UMat> undist_xmap_vector;
  std::vector<cv::UMat> undist_ymap_vector;

  undist_xmap_vector_ = std::vector<cv::UMat>(num_img_);
  undist_ymap_vector_ = std::vector<cv::UMat>(num_img_);
  
  InitUndistortMap();

  // for (size_t i = 0; i < num_img_; i++) {
  //     cv::Size sz = image_vector_[i].size();
  //     cv::Mat map_x(sz, CV_32FC1);
  //     cv::Mat map_y(sz, CV_32FC1);
  //     for (int y = 0; y < sz.height; y++) {
  //         for (int x = 0; x < sz.width; x++) {
  //             map_x.at<float>(y, x) = static_cast<float>(x);
  //             map_y.at<float>(y, x) = static_cast<float>(y);
  //         }
  //     }
  //     map_x.copyTo(undist_xmap_vector_[i]);
  //     map_y.copyTo(undist_ymap_vector_[i]);
  // }

  for (size_t img_idx = 0; img_idx < num_img_; ++img_idx) {
    cv::remap(image_vector_[img_idx],
              image_vector_[img_idx],
              undist_xmap_vector_[img_idx],
              undist_ymap_vector_[img_idx],
              cv::INTER_LINEAR);
  }
  #ifdef debug
  for(int i = 0; i < image_vector_.size(); i++) {
    cv::imwrite("../res/remap_"+std::to_string(i)+".jpg", image_vector_[i]);
  }
  #endif


  InitCameraParam();
  InitWarper();

  std::cout << "[StitchingParamGenerator] Initialized." << std::endl;
}

  std::vector<ImageFeatures> makeFeatures(
      const Eigen::Matrix<double, 259, Eigen::Dynamic>& featMat)
  {
      ImageFeatures feats;
      int num_points = featMat.cols();

      feats.keypoints.reserve(num_points);
      feats.descriptors.create(num_points, 256, CV_32F); // 每个点256维

      for (int i = 0; i < num_points; ++i)
      {
          double score = static_cast<double>(featMat(0, i));
          double x = static_cast<double>(featMat(1, i));
          double y = static_cast<double>(featMat(2, i));

          // 添加关键点
          feats.keypoints.emplace_back(cv::KeyPoint(x, y, 8, -1, score));
      }
      Eigen::MatrixXf descEigen = featMat.block(3, 0, featMat.rows() - 3, featMat.cols()).cast<float>();

      cv::Mat descMat(descEigen.rows(), descEigen.cols(), CV_32F, (void*)descEigen.data());
      descMat = descMat.t();
      descMat = descMat.clone();  // Eigen 列主序转为行主序
      descMat.copyTo(feats.descriptors);

      return {feats};
  }

void StitchingParamGenerator::InitCameraParam() {
  // Ptr<Feature2D> finder;
  // finder = SIFT::create();
  std::vector<ImageFeatures> features(num_img_);
  // std::vector<Size> full_img_sizes(num_img_);
  // for (int i = 0; i < num_img_; ++i) {
  //   computeImageFeatures(finder, image_vector_[i], features[i]);
  //   features[i].img_idx = i;
  //   LOGLN("Features in image #" << i + 1 << ": " << features[i].keypoints.size());
  // }
  
  // LOGLN("Pairwise matching");
  std::vector<MatchesInfo> pairwise_matches;
  // // Ptr<FeaturesMatcher> matcher;
  // // if (matcher_type == "affine")
  // //   matcher = makePtr<AffineBestOf2NearestMatcher>(false, try_cuda, match_conf);
  // // else if (range_width == -1)
  // //   matcher = makePtr<BestOf2NearestMatcher>(try_cuda, match_conf);
  // // else
  // //   matcher = makePtr<BestOf2NearestRangeMatcher>(range_width, try_cuda,
  // //                                                 match_conf);
  // // (*matcher)(features, pairwise_matches);
  // // matcher->collectGarbage();


  // // 1. 创建BFMatcher（SIFT 用 NORM_L2）
  // cv::BFMatcher matcher2(cv::NORM_L2, false);

  // // 2. knn匹配，k=2 用于ratio test
  // std::vector<std::vector<cv::DMatch>> knn_matches;
  // matcher2.knnMatch(features[0].descriptors, features[1].descriptors, knn_matches, 2);

  // // 3. Lowe’s ratio test
  // const float ratio_thresh = 0.75f;
  // std::vector<cv::DMatch> good_matches;

  // for (auto& m : knn_matches) {
  //     if (m.size() == 2 && m[0].distance < ratio_thresh * m[1].distance) {
  //         good_matches.push_back(m[0]);
  //     }
  // }

  // // 4. 可选：根据距离再过滤一次
  // std::sort(good_matches.begin(), good_matches.end(),
  //           [](const cv::DMatch& a, const cv::DMatch& b) { return a.distance < b.distance; });
  // if (good_matches.size() > 1000)  // 防止太多
  //     good_matches.resize(1000);
  

  //使用superpoint

  // read config from file
  Configs configs("/home/ld/project/SuperPoint-SuperGlue-TensorRT/config/config.yaml", "/home/ld/project/SuperPoint-SuperGlue-TensorRT/weights/");
  // image_vector_[0] = cv::imread("/home/ld/project/gpu-based-image-stitching/datasets/air-4cam-mp4/origin1.jpg",cv::IMREAD_GRAYSCALE);
  // image_vector_[1] = cv::imread("/home/ld/project/gpu-based-image-stitching/datasets/air-4cam-mp4/origin2.jpg",cv::IMREAD_GRAYSCALE);
  
  // create superpoint detector and superglue matcher
  auto superpoint = std::make_shared<SuperPoint>(configs.superpoint_config);
  auto superglue = std::make_shared<SuperGlue>(configs.superglue_config);

  // build engine
  superpoint->build();
  superglue->build();

  // infer superpoint
  Eigen::Matrix<double, 259, Eigen::Dynamic> feature_points0, feature_points1;
  cv::Mat img1_gray,img2_gray;
  cv::Mat resize_img1,resize_img2;
  cv::Size size(640,480);
  float x_scale = image_vector_[0].cols / (float) size.width;
  float y_scale = image_vector_[0].rows / (float) size.height;
  cv::cvtColor(image_vector_[0], img1_gray, cv::COLOR_BGR2GRAY);
  cv::cvtColor(image_vector_[1], img2_gray, cv::COLOR_BGR2GRAY);

  cv::resize(img1_gray, resize_img1, size);
  cv::resize(img2_gray, resize_img2, size);


  superpoint->infer(resize_img1, feature_points0);
  superpoint->infer(resize_img2, feature_points1);

  // infer superglue
  std::vector<cv::DMatch> good_matches;
  superglue->matching_points(feature_points0, feature_points1, good_matches);
  
  #ifdef debug
  cv::Mat match_image;
  std::vector<cv::KeyPoint> keypoints0, keypoints1;
  for(size_t i = 0; i < feature_points0.cols(); ++i){
    double score = feature_points0(0, i);
    feature_points0(1, i)  = feature_points0(1, i) * x_scale;
    feature_points0(2, i) = feature_points0(2, i) * y_scale;
    keypoints0.emplace_back(feature_points0(1, i), feature_points0(2, i), 8, -1, score);
  }
  for(size_t i = 0; i < feature_points1.cols(); ++i){
    double score = feature_points1(0, i);
    feature_points1(1, i) = feature_points1(1, i) * x_scale;
    feature_points1(2, i) = feature_points1(2, i) * y_scale;
    keypoints1.emplace_back(feature_points0(1, i), feature_points0(2, i), 8, -1, score);
  }
  cv::drawMatches(image_vector_[0], keypoints0, image_vector_[1], keypoints1, good_matches, match_image);
  cv::imwrite("../res/superpoint.jpg",  match_image);
  
  #endif
  features[0] = makeFeatures(feature_points0)[0];
  features[1] = makeFeatures(feature_points1)[0];
  
  features[0].img_idx = 0;
  features[1].img_idx = 1;

  features[0].img_size = image_vector_[0].size();
  features[1].img_size = image_vector_[1].size();



  pairwise_matches.clear();
  pairwise_matches.resize(4);

  pairwise_matches[1] = makeMatchesInfo(0, 1, features[0], features[1], good_matches);

// 5. 可视化
// cv::Mat img_matches;
// cv::drawMatches(img1, kp1, img2, kp2, good_matches, img_matches,
//                 cv::Scalar::all(-1), cv::Scalar::all(-1),
//                 std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
// cv::imshow("BFMatcher Matches", img_matches);
// cv::waitKey();

  const MatchesInfo& m = pairwise_matches[1];
  int idx1 = m.src_img_idx;
  int idx2 = m.dst_img_idx;
  cv::Mat img1 = image_vector_[idx1]; // 原图数组
  cv::Mat img2 = image_vector_[idx2];

  // 画匹配线
  cv::Mat match_img;
  cv::drawMatches(img1, features[idx1].keypoints,
                  img2, features[idx2].keypoints,
                  m.matches, match_img);

  std::string winname = "../res/Matches_" + std::to_string(idx1) + "-" + std::to_string(idx2)+".jpg";
  cv::imwrite(winname, match_img);
   

  // Check if we should save matches graph
  if (save_graph) {
    LOGLN("Saving matches graph...");
    ofstream f(save_graph_to.c_str());
    f << matchesGraphAsString(img_names, pairwise_matches, conf_thresh);
  }
  Ptr<Estimator> estimator;
  if (estimator_type == "affine")
    estimator = makePtr<AffineBasedEstimator>();
  else
    estimator = makePtr<HomographyBasedEstimator>();
  if (!(*estimator)(features, pairwise_matches, camera_params_vector_)) {
    std::cout << "Homography estimation failed.\n";
    assert(false);
  }
  for (auto& i : camera_params_vector_) {
    Mat R;
    i.R.convertTo(R, CV_32F);
    i.R = R;
  }
  Ptr<detail::BundleAdjusterBase> adjuster;
  if (ba_cost_func == "reproj")
    adjuster = makePtr<detail::BundleAdjusterReproj>();
  else if (ba_cost_func == "ray")
    adjuster = makePtr<detail::BundleAdjusterRay>();
  else if (ba_cost_func == "affine")
    adjuster =
        makePtr<detail::BundleAdjusterAffinePartial>();
  else if (ba_cost_func == "no") adjuster = makePtr<NoBundleAdjuster>();
  else {
    std::cout << "Unknown bundle adjustment cost function: '"
              << ba_cost_func
              << "'.\n";
    assert(false);
  }
  adjuster->setConfThresh(conf_thresh);
  Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
  if (ba_refine_mask[0] == 'x') refine_mask(0, 0) = 1;
  if (ba_refine_mask[1] == 'x') refine_mask(0, 1) = 1;
  if (ba_refine_mask[2] == 'x') refine_mask(0, 2) = 1;
  if (ba_refine_mask[3] == 'x') refine_mask(1, 1) = 1;
  if (ba_refine_mask[4] == 'x') refine_mask(1, 2) = 1;
  adjuster->setRefinementMask(refine_mask);
  if (!(*adjuster)(features, pairwise_matches, camera_params_vector_)) {
    std::cout << "Camera parameters adjusting failed.\n";
    assert(false);
  }

  // std::vector<Mat> rmats;
  // for (auto& i : camera_params_vector_)
  //   rmats.push_back(i.R.clone());
  // waveCorrect(rmats, wave_correct);
  // for (size_t i = 0; i < camera_params_vector_.size(); ++i) {
  //   camera_params_vector_[i].R = rmats[i];
  //   LOGLN("Initial camera intrinsics #"
  //             << i + 1 << ":\nK:\n"
  //             << camera_params_vector_[i].K()
  //             << "\nR:\n" << camera_params_vector_[i].R);
  // }
}

void StitchingParamGenerator::InitWarper() {

  std::vector<double> focals;
  float median_focal_length;
  reproj_xmap_vector_ = std::vector<UMat>(num_img_);

  for (size_t i = 0; i < camera_params_vector_.size(); ++i) {
    LOGLN("Camera #" << i + 1 << ":\nK:\n" << camera_params_vector_[i].K()
                     << "\nR:\n" << camera_params_vector_[i].R);
    focals.push_back(camera_params_vector_[i].focal);
  }
  sort(focals.begin(), focals.end());
  if (focals.size() % 2 == 1)
    median_focal_length = static_cast<float>(focals[focals.size() / 2]);
  else
    median_focal_length =
        static_cast<float>(focals[focals.size() / 2 - 1] +
            focals[focals.size() / 2]) * 0.5f;

  Ptr<WarperCreator> warper_creator;
#ifdef HAVE_OPENCV_CUDAWARPING
  if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0) {
    if (warp_type == "plane")
      warper_creator = makePtr<cv::PlaneWarperGpu>();
    else if (warp_type == "cylindrical")
      warper_creator = makePtr<cv::CylindricalWarperGpu>();
    else if (warp_type == "spherical")
      warper_creator = makePtr<cv::SphericalWarperGpu>();
  } else
#endif
  {
    if (warp_type == "plane")
      warper_creator = makePtr<cv::PlaneWarper>();
    else if (warp_type == "affine")
      warper_creator = makePtr<cv::AffineWarper>();
    else if (warp_type == "cylindrical")
      warper_creator = makePtr<cv::CylindricalWarper>();
    else if (warp_type == "spherical")
      warper_creator = makePtr<cv::SphericalWarper>();
    else if (warp_type == "fisheye")
      warper_creator = makePtr<cv::FisheyeWarper>();
    else if (warp_type == "stereographic")
      warper_creator = makePtr<cv::StereographicWarper>();
    else if (warp_type == "compressedPlaneA2B1")
      warper_creator = makePtr<cv::CompressedRectilinearWarper>(2.0f, 1.0f);
    else if (warp_type == "compressedPlaneA1.5B1")
      warper_creator = makePtr<cv::CompressedRectilinearWarper>(1.5f, 1.0f);
    else if (warp_type == "compressedPlanePortraitA2B1")
      warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(2.0f, 1.0f);
    else if (warp_type == "compressedPlanePortraitA1.5B1")
      warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(1.5f, 1.0f);
    else if (warp_type == "paniniA2B1")
      warper_creator = makePtr<cv::PaniniWarper>(2.0f, 1.0f);
    else if (warp_type == "paniniA1.5B1")
      warper_creator = makePtr<cv::PaniniWarper>(1.5f, 1.0f);
    else if (warp_type == "paniniPortraitA2B1")
      warper_creator = makePtr<cv::PaniniPortraitWarper>(2.0f, 1.0f);
    else if (warp_type == "paniniPortraitA1.5B1")
      warper_creator = makePtr<cv::PaniniPortraitWarper>(1.5f, 1.0f);
    else if (warp_type == "mercator")
      warper_creator = makePtr<cv::MercatorWarper>();
    else if (warp_type == "transverseMercator")
      warper_creator = makePtr<cv::TransverseMercatorWarper>();
  }
  if (!warper_creator) {
    std::cout << "Can't create the following warper '" << warp_type << "'\n";
    assert(false);
  }
  rotation_warper_ =
      warper_creator->create(static_cast<float>(median_focal_length));
  LOGLN("warped_image_scale: " << median_focal_length);

  std::vector<cv::Point> image_point_vect(num_img_);

  for (int img_idx = 0; img_idx < num_img_; ++img_idx) {
    Mat_<float> K;
    camera_params_vector_[img_idx].K().convertTo(K, CV_32F);
    Rect rect = rotation_warper_->buildMaps(image_size_vector_[img_idx], K,
                                       camera_params_vector_[img_idx].R,
                                       reproj_xmap_vector_[img_idx],
                                       reproj_ymap_vector_[img_idx]);
    Point point(rect.x, rect.y);
    image_point_vect[img_idx] = point;
  }


  // Prepare images masks
  for (int img_idx = 0; img_idx < num_img_; ++img_idx) {
    mask_vector_[img_idx].create(image_vector_[img_idx].size(), CV_8U);
    mask_vector_[img_idx].setTo(Scalar::all(255));
    remap(mask_vector_[img_idx],
          mask_warped_vector_[img_idx],
          reproj_xmap_vector_[img_idx],
          reproj_ymap_vector_[img_idx],
          INTER_NEAREST);
    image_warped_size_vector_[img_idx] = mask_warped_vector_[img_idx].size();
  }

  timelapser_ = Timelapser::createDefault(timelapse_type);
  blender_ = Blender::createDefault(Blender::NO);
  timelapser_->initialize(image_point_vect, image_size_vector_);
  blender_->prepare(image_point_vect, image_size_vector_);

  std::vector<cv::Rect> projected_image_roi_vect = std::vector<cv::Rect>(num_img_);

  // Update corners and sizes
  // TODO(duchengyao): Figure out what bias means.
  Point roi_tl_bias(999999, 999999);
  for (int i = 0; i < num_img_; ++i) {
    // Update corner and size
    Size sz = image_vector_[i].size();
    Mat K;
    camera_params_vector_[i].K().convertTo(K, CV_32F);
    Rect roi = rotation_warper_->warpRoi(sz, K, camera_params_vector_[i].R);
    std::cout << "roi" << roi << std::endl;
    roi_tl_bias.x = min(roi.tl().x, roi_tl_bias.x);
    roi_tl_bias.y = min(roi.tl().y, roi_tl_bias.y);
    projected_image_roi_vect[i] = roi;
  }
  full_image_size_ = Point(0, 0);
  Point y_range = Point(-9999999, 999999);
  for (int i = 0; i < num_img_; ++i) {
    projected_image_roi_vect[i] -= roi_tl_bias;
    Point tl = projected_image_roi_vect[i].tl();
    Point br = projected_image_roi_vect[i].br();

    full_image_size_.x = max(br.x, full_image_size_.x);
    full_image_size_.y = max(br.y, full_image_size_.y);
    y_range.x = max(y_range.x, tl.y);
    y_range.y = min(y_range.y, br.y);
  }
  for (int i = 0; i < num_img_; ++i) {
    Rect rect = projected_image_roi_vect[i];
    rect.height =
        rect.height - (rect.br().y - y_range.y + y_range.x - rect.tl().y);
    rect.y = y_range.x - rect.y;
    projected_image_roi_vect[i] = rect;
    projected_image_roi_refined_vect_[i] = rect;
  }

  for (int i = 0; i < num_img_ - 1; ++i) {

    Rect rect_left = projected_image_roi_refined_vect_[i];
    int offset = (projected_image_roi_vect[i].br().x -
        projected_image_roi_vect[i + 1].tl().x) / 2;
    rect_left.width -= offset;
    Rect rect_right = projected_image_roi_vect[i + 1];
    rect_right.width -= offset;
    rect_right.x = offset;
    projected_image_roi_refined_vect_[i] = rect_left;
    projected_image_roi_refined_vect_[i + 1] = rect_right;
  }
}

void StitchingParamGenerator::InitUndistortMap() {
  std::vector<double> cam_focal_vector(num_img_);

  std::vector<cv::UMat> r_vector(num_img_);
  std::vector<cv::UMat> k_vector(num_img_);
  std::vector<std::vector<double>> d_vector(num_img_);
  cv::Size resolution;

  undist_xmap_vector_ = std::vector<cv::UMat>(num_img_);
  undist_ymap_vector_ = std::vector<cv::UMat>(num_img_);

  for (size_t i = 0; i < num_img_; i++) {
    cv::FileStorage fs_read(
        "../params/camchain_" + std::to_string(i) + ".yaml",

        cv::FileStorage::READ);
    if (!fs_read.isOpened()) {
      fprintf(stderr, "%s:%d:loadParams falied. 'camera.yml' does not exist\n", __FILE__, __LINE__);
      return;
    }
    cv::Mat R, K;
    fs_read["KMat"] >> K;
    K.copyTo(k_vector[i]);
    fs_read["D"] >> d_vector[i];
    fs_read["RMat"] >> R;
    R.copyTo(r_vector[i]);
    fs_read["focal"] >> cam_focal_vector[i];
    fs_read["resolution"] >> resolution;
  }

  for (size_t i = 0; i < num_img_; i++) {
    cv::UMat K;
    cv::UMat R;
    cv::UMat NONE;
    k_vector[i].convertTo(K, CV_32F);
    cv::UMat::eye(3, 3, CV_32F).convertTo(R, CV_32F);

    cv::initUndistortRectifyMap(
        K, d_vector[i], R, NONE, resolution,
        CV_32FC1, undist_xmap_vector_[i], undist_ymap_vector_[i]);
  }
}

void StitchingParamGenerator::GetReprojParams(
    std::vector<cv::UMat>& undist_xmap_vector,
    std::vector<cv::UMat>& undist_ymap_vector,
    std::vector<cv::UMat>& reproj_xmap_vector,
    std::vector<cv::UMat>& reproj_ymap_vector,
    std::vector<cv::Rect>& projected_image_roi_refined_vect) {

  undist_xmap_vector = undist_xmap_vector_;
  undist_ymap_vector = undist_ymap_vector_;
  reproj_xmap_vector = reproj_xmap_vector_;
  reproj_ymap_vector = reproj_ymap_vector_;
  projected_image_roi_refined_vect = projected_image_roi_refined_vect_;
  std::cout << "[GetReprojParams] projected_image_roi_vect_refined: " << std::endl;

  size_t i = 0;
  for (auto& roi : projected_image_roi_refined_vect) {
    std::cout << "[GetReprojParams] roi [" << i << ": "
              << roi.width << "x"
              << roi.height << " from ("
              << roi.x << ", "
              << roi.y << ")]" << std::endl;
    i++;
    if (roi.width < 0 || roi.height < 0 || roi.x < 0 || roi.y < 0) {
      std::cout << "StitchingParamGenerator did not find a suitable feature point under the current parameters, "
                << "resulting in an incorrect ROI. "
                << "Please use \"opencv/stitching_detailed\" to find the correct parameters. "
                << "(see https://docs.opencv.org/4.8.0/d8/d19/tutorial_stitcher.html)" << std::endl;
      assert (false);
    }
  }
}