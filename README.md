# Hybrid ORB + Optical Flow + PCA Rotation Tracker

A real-time computer vision system that tracks multiple objects in video streams and estimates their rotation axis and angular velocity using feature clustering and optical flow analysis.

## Overview

This tracker combines ORB feature detection, Lucas-Kanade optical flow, DBSCAN clustering, and PCA-based motion analysis to:
- Detect and track multiple objects simultaneously
- Estimate rotation axis (3D vector) for each object
- Calculate angular velocity (omega) in degrees per second
- Maintain object identity across frames

## Requirements
```bash
pip install opencv-python numpy scikit-learn
```

## Features

- **Multi-object tracking**: Automatically detects and tracks multiple moving objects
- **Rotation estimation**: Computes rotation axis and angular velocity using PCA on optical flow vectors
- **Temporal smoothing**: Applies exponential smoothing to angular velocity for stable measurements
- **Visual feedback**: Real-time display with bounding boxes, object IDs, and rotation parameters

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `VIDEO_SRC` | 0 | Video source (0 for webcam, or path to video file) |
| `MIN_FEATURES_PER_OBJECT` | 10 | Minimum ORB features required to detect an object |
| `DBSCAN_EPS` | 50 | Maximum distance between points in a cluster |
| `DBSCAN_MIN_SAMPLES` | 5 | Minimum points to form a cluster |
| `MAX_CENTROID_DIST` | 100 | Maximum distance to match objects between frames |
| `SMOOTHING_ALPHA` | 0.3 | Smoothing factor for angular velocity (0-1) |

## Usage
```bash
python rotation_tracker.py
```

Press **ESC** to exit the application.

## How It Works

1. **Feature Detection**: Extracts up to 2000 ORB keypoints from each frame
2. **Optical Flow**: Tracks feature points between consecutive frames using Lucas-Kanade method
3. **Clustering**: Groups nearby points using DBSCAN to identify distinct objects
4. **Object Matching**: Associates clusters with tracked objects based on centroid proximity
5. **Rotation Analysis**: 
   - Computes motion vectors for each cluster
   - Applies PCA to find principal motion direction (rotation axis)
   - Estimates angular velocity from motion magnitude
6. **Display**: Renders bounding boxes, object IDs, and rotation parameters on video

## Output Information

For each tracked object, the display shows:
- **Bounding box** (green rectangle)
- **Object ID** (top-left, blue text)
- **Angular velocity** in deg/s (below ID, cyan text)
- **Rotation axis** as 3D vector (bottom, yellow text)
- **Feature points** (red dots)

## Limitations

- Requires sufficient texture/features on objects
- Rotation axis estimation assumes planar motion
- Performance depends on lighting conditions and video quality
- May lose tracking during rapid movements or occlusions

## Customization

Adjust parameters at the top of the script to tune for your specific use case:
- Increase `DBSCAN_EPS` for larger objects
- Decrease `SMOOTHING_ALPHA` for more responsive but noisier measurements
- Modify camera resolution via `cap.set(3, width)` and `cap.set(4, height)`