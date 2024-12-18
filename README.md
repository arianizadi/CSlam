# SLAM 3D Map Full Checklist

## Preprocessing
- [x] **Read Video**: Implemented using `cv::VideoCapture` to load video frames sequentially.
- [x] **Grayscale Conversion**: Implemented using `cv::cvtColor` to convert frames to grayscale.
- [x] **Gaussian Blur**: Applied Gaussian smoothing to reduce noise in frames.
- [x] **Histogram Equalization**: Enhanced contrast in frames using histogram equalization.
- [ ] **Synchronize Cameras**: If using multiple cameras, ensure temporal synchronization of frames.

## Feature Detection and Matching
- [x] **Feature Extraction**: Used ORB to detect keypoints and descriptors in frames.
- [x] **Feature Matching**: Implemented using `BFMatcher` with the Hamming norm.
- [x] **Feature Selection**:
  - [x] Apply Lowe’s ratio test to retain reliable matches.
  - [x] Filter matches using geometric constraints like the Fundamental Matrix with RANSAC.

## Pose Estimation
- [ ] **Camera Calibration**:
  - [x] Load or define intrinsic parameters (focal length, principal point, distortion coefficients).
  - [ ] Estimate intrinsics using calibration patterns or auto-calibration if unknown.
- [ ] **Essential Matrix**: Compute the Essential Matrix (`cv::findEssentialMat`) using inlier matches.
- [ ] **Recover Pose**: Decompose the Essential Matrix into rotation (R) and translation (t) (`cv::recoverPose`).
- [ ] **Validate Pose**:
  - [ ] Check for degenerate cases (e.g., low parallax).
  - [ ] Handle scale ambiguity in monocular SLAM.
- [ ] **Accumulate Poses**: Maintain global camera pose by chaining relative poses.

## 3D Point Triangulation
- [ ] **Triangulate Points**: Use `cv::triangulatePoints` to reconstruct 3D points from inlier matches and camera matrices.
- [ ] **Point Filtering**:
  - [ ] Remove outliers based on reprojection error.
  - [ ] Discard points with negative depth values.
- [ ] **Store 3D Points**: Add reliable triangulated points to the global map.

## Multi-Camera Integration
- [ ] **Extrinsic Calibration**: Determine relative positions and orientations of cameras.
- [ ] **Camera Stitching**:
  - [ ] Align feature points from overlapping views.
  - [ ] Merge triangulated 3D points into a unified map.
- [ ] **Handle Overlaps**: Remove duplicate points in overlapping regions based on proximity and reprojection.

## Map Optimization (g2o)
- [ ] **Bundle Adjustment**:
  - [ ] Initialize `g2o` with 3D points, camera poses, and observations (keypoint matches).
  - [ ] Optimize the entire map to refine poses and 3D points.
- [ ] **Add Loop Closure**:
  - [ ] Detect revisited areas using Bag-of-Words or other methods.
  - [ ] Add loop-closure constraints to the optimization problem.
- [ ] **Dynamic Object Handling**:
  - [ ] Identify and discard points on dynamic objects (e.g., moving cars or pedestrians).

## Visualization (Pangolin)
- [ ] **Integrate Pangolin**:
  - [ ] Visualize 3D trajectory and point cloud in real time.
  - [ ] Continuously update the global camera pose and 3D map during processing.
- [ ] **Colored Point Cloud**:
  - [ ] Associate color data from frames with 3D points.
- [ ] **Trajectory Plot**:
  - [ ] Visualize the camera’s trajectory alongside the point cloud.

## Debugging and Testing
- [ ] **Debug Pose Estimation**:
  - [ ] Visualize the camera trajectory to ensure poses are realistic.
  - [ ] Check for errors like inconsistent rotation or translation.
- [ ] **Debug 3D Points**: Validate the reconstructed map for issues like points behind the camera.
- [ ] **Synthetic Dataset Testing**:
  - [ ] Test with datasets like TUM, KITTI, or EuRoC to benchmark accuracy.
- [ ] **Edge Case Handling**:
  - [ ] Handle poor lighting, blank walls, or rapid motion.

## Optional Enhancements
- [ ] **Real-Time SLAM**:
  - [ ] Use a local map for efficiency and update the global map less frequently.
- [ ] **Scale Estimation**:
  - [ ] Use prior knowledge, stereo vision, or object dimensions to resolve monocular scale ambiguity.
- [ ] **Optimization Framework Alternatives**:
  - [ ] Explore alternatives like Ceres Solver or Sophus.
- [ ] **Multi-View Constraints**:
  - [ ] Add support for multi-view geometry for complex camera paths.

## Final Integration
- Combine all modules:
  - [ ] Frame processing pipeline (feature matching, pose estimation, triangulation).
  - [ ] Multi-camera stitching.
  - [ ] Map optimization (g2o).
  - [ ] Visualization (Pangolin).

---

### Output
- A 3D map (point cloud) of the scene.
- A unified camera trajectory visualized in real time with Pangolin.
