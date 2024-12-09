#include <chrono>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

using namespace cv;
using std::cout, std::endl, std::cerr;
using std::vector;

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

  Mat frame, grayFrame;
  int maxCorners = 100;
  double qualityLevel = 0.01;
  double minDistance = 10.0;

  while(cap.read(frame)) {
    start = std::chrono::high_resolution_clock::now();

    cvtColor(frame, grayFrame, COLOR_BGR2GRAY);

    Ptr< ORB > orb = ORB::create();
    vector< KeyPoint > keypoints;
    Mat descriptors;

    orb->detectAndCompute(grayFrame, noArray(), keypoints, descriptors);
    drawKeypoints(frame, keypoints, frame);

    end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration< double >(end - start).count();
    fps = 1.0 / elapsed;

    cv::putText(frame,
                "FPS: " + std::to_string(fps),
                cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX,
                1,
                Scalar(0, 255, 0),
                2);

    imshow("Live", frame);

    if(waitKey(5) >= 0) {
      break;
    }
  }

  return 0;
}
