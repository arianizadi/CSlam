#include <chrono>
#include <climits>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/viz.hpp>

using namespace cv;
using std::cerr, std::vector, std::cout, std::endl;

Mat calculateIntrinsics(Mat& frame) {
  double fx = frame.cols;
  double fy = fx;
  double cx = frame.cols / 2.0;
  double cy = frame.rows / 2.0;

  Mat K = (Mat_< double >(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
  return K;
}

void processFrameAndUpdatePose(
    Mat& frame1, Mat& frame2, Mat& R, Mat& t, Mat& K) {
  Mat gray1, gray2;
  cvtColor(frame1, gray1, COLOR_BGR2GRAY);
  cvtColor(frame2, gray2, COLOR_BGR2GRAY);

  GaussianBlur(gray1, gray1, Size(3, 3), 0);
  GaussianBlur(gray2, gray2, Size(3, 3), 0);
  equalizeHist(gray1, gray1);
  equalizeHist(gray2, gray2);

  Ptr< ORB > orb = ORB::create();
  orb->setMaxFeatures(1000);

  vector< KeyPoint > keypoints1, keypoints2;
  Mat descriptor1, descriptor2;

  orb->detectAndCompute(gray1, noArray(), keypoints1, descriptor1);
  orb->detectAndCompute(gray2, noArray(), keypoints2, descriptor2);

  if(keypoints1.size() < 8 || keypoints2.size() < 8) {
    cerr << "Not enough keypoints. Skipping frame." << endl;
    return;
  }

  BFMatcher matcher(NORM_HAMMING);
  vector< vector< DMatch > > knnMatches;
  matcher.knnMatch(descriptor1, descriptor2, knnMatches, 2);

  vector< DMatch > goodMatches;
  const float ratio = 0.7F;
  for(const auto& knnMatch : knnMatches) {
    if(knnMatch.size() >= 2
       && knnMatch[0].distance < ratio * knnMatch[1].distance) {
      goodMatches.push_back(knnMatch[0]);
    }
  }

  if(goodMatches.size() < 8) {
    cerr << "Not enough good matches. Skipping frame." << endl;
    return;
  }

  vector< Point2f > points1, points2;
  for(const auto& match : goodMatches) {
    points1.push_back(keypoints1[match.queryIdx].pt);
    points2.push_back(keypoints2[match.trainIdx].pt);
  }

  // Calculate Essential Matrix
  Mat E = findEssentialMat(points1, points2, K, RANSAC, 0.999, 1.0);

  if(E.empty()) {
    cerr << "Failed to compute essential matrix. Skipping frame." << endl;
    return;
  }

  // Recover R and t from Essential Matrix
  Mat tempR1, tempR2, tempT;
  decomposeEssentialMat(E, tempR1, tempR2, tempT);

  // Use recoverPose to get the correct R and t
  Mat mask;
  int inliers = recoverPose(E, points1, points2, K, R, t, mask);
}

int main(int argc, char** argv) {
  if(argc != 2) {
    cerr << "Usage: " << argv[0] << " <video_path>" << endl;
    return -1;
  }

  std::string video_path = argv[1];
  VideoCapture cap(video_path);

  if(!cap.isOpened()) {
    cerr << "ERROR! Unable to open video: " << video_path << endl;
    return -1;
  }

  viz::Viz3d window("Camera Motion");
  window.setBackgroundColor(viz::Color::black());

  window.showWidget("Coordinate Widget", viz::WCoordinateSystem(0.1));

  viz::WCameraPosition cpw_frustum(Vec2f(0.9, 0.4), 0.15, viz::Color::yellow());
  viz::WCameraPosition cpw_axes(0.1);
  window.showWidget("CPW_FRUSTUM", cpw_frustum);
  window.showWidget("CPW_AXES", cpw_axes);

  window.showWidget("grid", viz::WGrid(Vec2i::all(5), Vec2d::all(0.5)));

  vector< Point3d > trajectory;
  window.setWindowSize(Size(1280, 720));
  window.setViewerPose(
      viz::makeCameraPose(Vec3d(0, -0.5, -2), Vec3d(0, 0, 0), Vec3d(0, -1, 0)));

  Mat currentFrame, previousFrame;
  cap >> previousFrame;

  Size frameDim(960, 540);
  resize(previousFrame, previousFrame, frameDim);

  // Initialize camera parameters
  Mat K = calculateIntrinsics(previousFrame);

  // Initialize camera pose
  Mat R = Mat::eye(3, 3, CV_64F);
  Mat t = Mat::zeros(3, 1, CV_64F);
  Mat currentR, currentT;

  // Accumulated transformation
  Mat accR = Mat::eye(3, 3, CV_64F);
  Mat accT = Mat::zeros(3, 1, CV_64F);

  double fps = 0;
  int frameCount = 0;
  std::chrono::time_point< std::chrono::high_resolution_clock > start, end;

  while(cap.read(currentFrame) && frameCount < INT_MAX) {
    start = std::chrono::high_resolution_clock::now();

    resize(currentFrame, currentFrame, frameDim);

    processFrameAndUpdatePose(
        previousFrame, currentFrame, currentR, currentT, K);

    if(!currentR.empty() && !currentT.empty()) {
      // Scale down the translation
      accT = accT + currentT * 0.01;

      // Accumulate rotation
      accR = currentR * accR;

      // Create new camera pose
      Mat pose = Mat::eye(4, 4, CV_64F);
      accR.copyTo(pose(Rect(0, 0, 3, 3)));
      accT.copyTo(pose(Rect(3, 0, 1, 3)));

      // Update camera widgets with new pose
      Affine3d cameraPose(pose);
      window.setWidgetPose("CPW_FRUSTUM", cameraPose);
      window.setWidgetPose("CPW_AXES", cameraPose);

      // Add point to trajectory
      trajectory.emplace_back(accT);
      if(trajectory.size() > 100) { // Keep only last 100 points
        trajectory.erase(trajectory.begin());
      }

      // Draw trajectory
      if(trajectory.size() > 1) {
        viz::WPolyLine trajectory_widget(trajectory, viz::Color::green());
        window.showWidget("TRAJECTORY", trajectory_widget);
      }

      // Add a sphere at current position
      viz::WSphere current_pos(Point3d(accT), 0.02, 10, viz::Color::red());
      window.showWidget("CURRENT_POS", current_pos);

      // Automatically adjust view to follow camera
      Vec3d cam_pos(accT);
      Vec3d viz_pos = cam_pos - Vec3d(0, -0.5, -2);
      window.setViewerPose(
          viz::makeCameraPose(viz_pos, cam_pos, Vec3d(0, -1, 0)));
      window.spinOnce(1, true);

      cout << "Camera Position: x=" << accT.at< double >(0)
           << " y=" << accT.at< double >(1) << " z=" << accT.at< double >(2)
           << endl;
    }

    end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration< double >(end - start).count();
    fps = 1.0 / elapsed;

    // Display original frame with FPS
    putText(currentFrame,
            "FPS: " + std::to_string(fps),
            Point(10, 30),
            FONT_HERSHEY_SIMPLEX,
            1,
            Scalar(0, 255, 0),
            2);

    imshow("Video Feed", currentFrame);

    int key = waitKey(1);
    if(key == 27) { // ESC
      break;
    }

    previousFrame = currentFrame.clone();
    frameCount++;
  }

  return 0;
}