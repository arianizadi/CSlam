#include <chrono>
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
using std::cout, std::endl, std::cerr, std::vector;

Mat processFrame(Mat& frame1, Mat& frame2) {
  Mat resFrame, gray1, gray2;

  cvtColor(frame1, gray1, COLOR_BGR2GRAY);
  cvtColor(frame2, gray2, COLOR_BGR2GRAY);

  Ptr< ORB > orb = orb->create();
  vector< KeyPoint > keypoints1, keypoints2;
  Mat descriptor1, descriptor2;

  orb->detectAndCompute(gray1, noArray(), keypoints1, descriptor1);
  orb->detectAndCompute(gray2, noArray(), keypoints2, descriptor2);

  BFMatcher bfMatcher;
  bfMatcher.create(NORM_HAMMING, true);

  vector< DMatch > matches;
  bfMatcher.match(descriptor1, descriptor2, matches);

  vector< DMatch > inlierMatches;
  vector< Point2f > points1, points2;
  for(const auto& match : matches) {
    points1.emplace_back(keypoints1[match.queryIdx].pt);
    points2.emplace_back(keypoints2[match.trainIdx].pt);
  }

  vector< uchar > inlierMask;
  findFundamentalMat(points1, points2, FM_RANSAC, 3, 0.99, inlierMask);
  for(int i = 0; i < inlierMask.size(); ++i) {
    if(inlierMask[i]) {
      inlierMatches.emplace_back(matches[i]);
    }
  }

  drawMatches(frame1, keypoints1, frame2, keypoints2, matches, resFrame);

  return resFrame;
}

int main() {
  std::string video_path
      = "/Users/arianizadi/Documents/Projects/Koshee/CSlam/data/drive2.webm";
  VideoCapture cap(video_path);

  if(!cap.isOpened()) {
    cerr << "ERROR! Unable to open video\n";
    return -1;
  }

  double fps = 0;
  std::chrono::time_point< std::chrono::high_resolution_clock > start, end;

  Mat currentFrame, previousFrame;

  cap >> previousFrame;

  while(cap.read(currentFrame)) {
    start = std::chrono::high_resolution_clock::now();

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

    if(waitKey(5) >= 0) {
      break;
    }

    previousFrame = currentFrame;
  }

  return 0;
}
