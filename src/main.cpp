#include <algorithm>
#include <chrono>
#include <climits>
#include <deque>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp> // For KLT tracker
#include <opencv2/videoio.hpp>
#include <opencv2/viz.hpp>

using namespace cv;
using std::cerr, std::vector, std::cout, std::endl;

// Camera path smoother class
class CameraPathSmoother {
private:
  const int bufferSize;
  std::deque< Affine3d > poses;
  Affine3d lastSmoothedPose;

public:
  CameraPathSmoother(int buffSize = 10) : bufferSize(buffSize) {
    lastSmoothedPose = Affine3d::Identity();
  }

  void addPose(const Mat& R, const Mat& t) {
    Mat Rt = Mat::eye(4, 4, CV_64F);
    R.copyTo(Rt(Rect(0, 0, 3, 3)));
    t.copyTo(Rt(Rect(3, 0, 1, 3)));

    Affine3d pose(Rt);
    poses.push_back(pose);

    if(poses.size() > bufferSize) {
      poses.pop_front();
    }
  }

  Affine3d getSmoothedPose() {
    if(poses.empty()) {
      return Affine3d::Identity();
    }

    if(poses.size() == 1) {
      lastSmoothedPose = poses[0];
      return poses[0];
    }

    // Simple weighted average for translations
    Vec3d avgTranslation(0, 0, 0);
    double weightSum = 0;

    for(size_t i = 0; i < poses.size(); i++) {
      double weight = i + 1; // More weight to recent poses
      avgTranslation += weight * poses[i].translation();
      weightSum += weight;
    }

    avgTranslation = avgTranslation / weightSum;

    // For rotation, simply use the most recent
    Mat R = Mat(poses.back().rotation());

    // Create new smoothed pose
    Affine3d smoothedPose = Affine3d(R, avgTranslation);

    // Temporal smoothing
    Vec3d newT = 0.8 * smoothedPose.translation()
                 + 0.2 * lastSmoothedPose.translation();
    Affine3d temporallySmoothedPose(R, newT);

    lastSmoothedPose = temporallySmoothedPose;
    return temporallySmoothedPose;
  }
};

Mat calculateIntrinsics(Mat& frame) {
  double fx = frame.cols;
  double fy = fx;
  double cx = frame.cols / 2.0;
  double cy = frame.rows / 2.0;

  Mat K = (Mat_< double >(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
  return K;
}

// Using KLT tracker instead of feature matching for better tracking in driving
// videos
bool trackFeatures(Mat& prevFrame,
                   Mat& currFrame,
                   vector< Point2f >& prevPoints,
                   vector< Point2f >& currPoints,
                   Mat& K,
                   Mat& R,
                   Mat& t) {
  // Convert to grayscale
  Mat prevGray, currGray;
  cvtColor(prevFrame, prevGray, COLOR_BGR2GRAY);
  cvtColor(currFrame, currGray, COLOR_BGR2GRAY);

  // Initialize status and error vectors
  vector< uchar > status;
  vector< float > err;

  // If no previous points, detect features to track
  if(prevPoints.empty()) {
    // Use FAST detector to find corners
    vector< KeyPoint > keypoints;
    int maxCorners = 3000;
    double qualityLevel = 0.01;
    double minDistance = 7;

    // Try FAST first
    FAST(prevGray, keypoints, 20, true);

    // If not enough points, try goodFeaturesToTrack
    if(keypoints.size() < 300) {
      goodFeaturesToTrack(
          prevGray, prevPoints, maxCorners, qualityLevel, minDistance);
    } else {
      KeyPoint::convert(keypoints, prevPoints);
    }

    // Refine corner locations
    if(!prevPoints.empty()) {
      cornerSubPix(
          prevGray,
          prevPoints,
          Size(10, 10),
          Size(-1, -1),
          TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03));
    }
  }

  // Early exit if no points to track
  if(prevPoints.empty() || prevPoints.size() < 10) {
    cerr << "Not enough points to track: "
         << (prevPoints.empty() ? 0 : prevPoints.size()) << endl;
    return false;
  }

  // Clear current points before tracking
  currPoints.clear();

  // Track points using Lucas-Kanade optical flow
  calcOpticalFlowPyrLK(
      prevGray,
      currGray,
      prevPoints,
      currPoints,
      status,
      err,
      Size(21, 21),
      3,
      TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 30, 0.01),
      OPTFLOW_LK_GET_MIN_EIGENVALS);

  // Filter tracked points using status
  vector< Point2f > good_prev, good_curr;
  for(size_t i = 0; i < status.size(); i++) {
    if(status[i]) {
      good_prev.push_back(prevPoints[i]);
      good_curr.push_back(currPoints[i]);
    }
  }

  // Check if we have enough points
  if(good_prev.size() < 10) {
    cerr << "Not enough good tracked points: " << good_prev.size() << endl;
    return false;
  }

  // Filter out outliers with RANSAC
  Mat mask;
  try {
    Mat E = findEssentialMat(good_prev, good_curr, K, RANSAC, 0.999, 1.0, mask);

    if(E.empty()) {
      cerr << "Failed to compute essential matrix" << endl;
      return false;
    }

    // Extract inliers
    vector< Point2f > inlier_prev, inlier_curr;
    for(size_t i = 0; i < mask.rows; i++) {
      if(mask.at< uchar >(i)) {
        inlier_prev.push_back(good_prev[i]);
        inlier_curr.push_back(good_curr[i]);
      }
    }

    // Update tracked points for next frame
    prevPoints = inlier_curr;

    // Check inlier count
    if(inlier_prev.size() < 8) {
      cerr << "Not enough inliers: " << inlier_prev.size() << endl;
      return false;
    }

    // Recover R and t from essential matrix
    int inliers = recoverPose(E, inlier_prev, inlier_curr, K, R, t);

    if(inliers < 8) {
      cerr << "Not enough inliers from pose recovery: " << inliers << endl;
      return false;
    }

    // Display tracked points
    Mat display = currFrame.clone();
    for(size_t i = 0; i < inlier_curr.size(); i++) {
      circle(display, inlier_curr[i], 3, Scalar(0, 255, 0), -1);
      line(display, inlier_prev[i], inlier_curr[i], Scalar(0, 0, 255), 1);
    }
    imshow("Tracked Points", display);

    return true;
  } catch(const cv::Exception& e) {
    cerr << "OpenCV exception: " << e.what() << endl;
    return false;
  }
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

  // Get video properties
  int frameWidth = cap.get(CAP_PROP_FRAME_WIDTH);
  int frameHeight = cap.get(CAP_PROP_FRAME_HEIGHT);
  double fps = cap.get(CAP_PROP_FPS);

  cout << "Video: " << frameWidth << "x" << frameHeight << " @ " << fps
       << " FPS" << endl;

  // Setup downsampling for faster processing
  Size frameDim(640, 360); // More aggressive downsampling for speed

  // Create visualization window
  viz::Viz3d window("Camera Motion");
  window.setBackgroundColor(viz::Color::black());
  window.showWidget("Coordinate Widget", viz::WCoordinateSystem(0.5));
  window.showWidget("grid", viz::WGrid(Vec2i::all(10), Vec2d::all(1.0)));

  // Camera widgets
  viz::WCameraPosition cpw_frustum(Vec2f(0.9, 0.6), 0.2, viz::Color::yellow());
  viz::WCameraPosition cpw_axes(0.2);
  window.showWidget("CPW_FRUSTUM", cpw_frustum);
  window.showWidget("CPW_AXES", cpw_axes);

  // Window settings
  window.setWindowSize(Size(800, 600));
  window.setViewerPose(
      viz::makeCameraPose(Vec3d(0, -2, -20), Vec3d(0, 0, 0), Vec3d(0, -1, 0)));

  // Read first frame
  Mat currentFrame, previousFrame;
  cap >> previousFrame;

  if(previousFrame.empty()) {
    cerr << "First frame is empty!" << endl;
    return -1;
  }

  // Resize for faster processing
  resize(previousFrame, previousFrame, frameDim);

  // Camera intrinsics
  Mat K = calculateIntrinsics(previousFrame);
  cout << "Camera matrix: " << K << endl;

  // Initialize tracking
  vector< Point2f > prevPoints; // Will be filled in trackFeatures
  vector< Point2f > currPoints;

  // Camera pose variables
  Mat R = Mat::eye(3, 3, CV_64F);
  Mat t = Mat::zeros(3, 1, CV_64F);

  // Accumulated pose
  Mat accR = Mat::eye(3, 3, CV_64F);
  Mat accT = Mat::zeros(3, 1, CV_64F);

  // Path smoothing
  CameraPathSmoother pathSmoother(15);
  vector< Point3d > trajectory;

  // Frame counters
  int frameCount = 0;
  int skipCount = 0;
  int successCount = 0;

  // Timing
  auto startTime = std::chrono::high_resolution_clock::now();

  // Enable skip ahead for driving videos
  int frameSkip = 1; // Process every nth frame

  // Process frames
  while(frameCount < INT_MAX) {
    // Skip frames if needed
    for(int i = 0; i < frameSkip && cap.read(currentFrame); i++) {
      // Just advance the video
    }

    // If we can't read any more frames, we're done
    if(!cap.read(currentFrame) || currentFrame.empty()) {
      break;
    }

    // Resize for faster processing
    resize(currentFrame, currentFrame, frameDim);

    // Track features
    Mat frame_R, frame_t;
    bool success = trackFeatures(previousFrame,
                                 currentFrame,
                                 prevPoints,
                                 currPoints,
                                 K,
                                 frame_R,
                                 frame_t);

    if(success) {
      successCount++;

      // Apply constant scale factor (adjust as needed for your video)
      double scale = 0.1; // Larger scale for driving videos

      // Scale the translation and accumulate
      accT = accT + scale * (accR * frame_t);

      // Accumulate rotation
      accR = frame_R * accR;

      // Add to path smoother
      pathSmoother.addPose(accR, accT);

      // Get smoothed pose for visualization
      Affine3d cameraPose = pathSmoother.getSmoothedPose();

      // Update visualization
      window.setWidgetPose("CPW_FRUSTUM", cameraPose);
      window.setWidgetPose("CPW_AXES", cameraPose);

      // Update trajectory
      Vec3d position = cameraPose.translation();
      trajectory.emplace_back(Point3d(position));

      // Keep trajectory manageable
      if(trajectory.size() > 200) {
        trajectory.erase(trajectory.begin());
      }

      // Draw trajectory
      if(trajectory.size() > 1) {
        viz::WPolyLine trajectoryWidget(trajectory, viz::Color::green());
        window.showWidget("TRAJECTORY", trajectoryWidget);
      }

      // Mark current position
      viz::WSphere currentPos(Point3d(position), 0.2, 10, viz::Color::red());
      window.showWidget("CURRENT_POS", currentPos);

      // Show position info
      cout << "Frame: " << frameCount << " | Pos: [" << position[0] << ", "
           << position[1] << ", " << position[2] << "]"
           << " | Success: " << successCount << "/" << frameCount << endl;

      // Reset skip counter on success
      skipCount = 0;
    } else {
      skipCount++;

      // If we're skipping too many frames, try to reinitialize
      if(skipCount > 5) {
        cout << "Reinitializing tracking after " << skipCount
             << " skipped frames" << endl;
        prevPoints.clear(); // Force redetection of features
        skipCount = 0;
      }
    }

    // Update viz window
    window.spinOnce(1, true);

    // Calculate elapsed time and FPS
    auto endTime = std::chrono::high_resolution_clock::now();
    double elapsed
        = std::chrono::duration< double >(endTime - startTime).count();
    double currentFps = 1.0 / elapsed;
    startTime = endTime;

    // Display info on frame
    putText(currentFrame,
            "FPS: " + std::to_string(int(currentFps))
                + " | Frame: " + std::to_string(frameCount)
                + " | Success: " + std::to_string(successCount) + "/"
                + std::to_string(frameCount),
            Point(10, 30),
            FONT_HERSHEY_SIMPLEX,
            0.7,
            Scalar(0, 255, 0),
            2);

    // Show frames
    imshow("Video Feed", currentFrame);

    // Check for user input
    int key = waitKey(1);
    if(key == 27) { // ESC to exit
      break;
    } else if(key == ' ') { // Space to pause
      waitKey(0);
    }

    // Update frames
    previousFrame = currentFrame.clone();
    frameCount++;
  }

  cout << "Video processing complete. Processed " << frameCount
       << " frames with " << successCount << " successful tracks ("
       << (successCount * 100.0 / frameCount) << "%)" << endl;

  waitKey(0);
  return 0;
}