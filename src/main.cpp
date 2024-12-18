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

Mat processFrame(Mat& frame1, Mat& frame2) {
  Mat resFrame, gray1, gray2;

  cvtColor(frame1, gray1, COLOR_BGR2GRAY);
  cvtColor(frame2, gray2, COLOR_BGR2GRAY);

  GaussianBlur(gray1, gray1, Size(3, 3), 0);
  GaussianBlur(gray2, gray2, Size(3, 3), 0);
  equalizeHist(gray1, gray1);
  equalizeHist(gray2, gray2);

  Mat K = calculateIntrinsics(gray1);

  Ptr< ORB > orb = orb->create();
  orb->setMaxFeatures(100);

  vector< KeyPoint > keypoints1, keypoints2;
  Mat descriptor1, descriptor2;

  orb->detectAndCompute(gray1, noArray(), keypoints1, descriptor1);
  orb->detectAndCompute(gray2, noArray(), keypoints2, descriptor2);

  if(keypoints1.size() < 5 || keypoints2.size() < 5 || descriptor1.empty()
     || descriptor2.empty()) {
    cerr << "Not enough keypoints or descriptors. Skipping frame." << endl;
    return frame1.clone();
  }

  BFMatcher bfMatcher(NORM_HAMMING);
  vector< vector< DMatch > > knnMatches;
  bfMatcher.knnMatch(descriptor1, descriptor2, knnMatches, 2);

  vector< DMatch > goodMatches;
  float ratio = 0.6;
  for(const auto& knnMatch : knnMatches) {
    if(knnMatch.size() >= 2
       && knnMatch[0].distance < ratio * knnMatch[1].distance) {
      goodMatches.push_back(knnMatch[0]);
    }
  }

  vector< Point2f > points1, points2;
  for(const auto& match : goodMatches) {
    points1.emplace_back(keypoints1[match.queryIdx].pt);
    points2.emplace_back(keypoints2[match.trainIdx].pt);
  }

  vector< uchar > inlierMask;
  if(!points1.empty() && !points2.empty()) {
    findFundamentalMat(points1, points2, FM_RANSAC, 3, 0.99, inlierMask);
  }

  Mat matchesImg;
  drawMatches(frame1,
              keypoints1,
              frame2,
              keypoints2,
              goodMatches,
              matchesImg,
              Scalar(0, 255, 0), // Color for good matches (Green)
              Scalar(0, 0, 255), // Color for single points (Red)
              vector< char >(),
              DrawMatchesFlags::DEFAULT
                  | DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

  return matchesImg;
}

int main() {
  std::string video_path
      = "/Users/arianizadi/Documents/Projects/Koshee/CSlam/data/snow.webm";
  VideoCapture cap(video_path);

  if(!cap.isOpened()) {
    cerr << "ERROR! Unable to open video\n";
    return -1;
  }

  double fps = 0;
  int frame = 0;
  std::chrono::time_point< std::chrono::high_resolution_clock > start, end;

  Mat currentFrame, previousFrame;
  cap >> previousFrame;

  Size frameDim = Size(960, 540);

  while(cap.read(currentFrame) && frame < INT_MAX) {
    start = std::chrono::high_resolution_clock::now();

    resize(currentFrame, currentFrame, frameDim);
    resize(previousFrame, previousFrame, frameDim);

    Mat resFrame = processFrame(previousFrame, currentFrame);

    end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration< double >(end - start).count();
    fps = 1.0 / elapsed;

    cv::putText(resFrame,
                "FPS: " + std::to_string(fps),
                cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX,
                1,
                Scalar(0, 255, 0),
                2);

    imshow("Live", resFrame);

    // Escape to exit
    // Space to toggle pause
    int spaceKey = 32;
    int escapeKey = 27;

    int key = waitKey(5);
    if(key == escapeKey) {
      break;
    } else if(key == spaceKey) {
      while(true) {
        int key2 = waitKey(5);
        if(key2 == spaceKey) {
          break;
        } else if(key2 == escapeKey) {
          return 0;
        }
      }
    }

    previousFrame = currentFrame;

    ++frame;
  }

  return 0;
}
