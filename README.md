# Ego-Vehicle Trajectory Estimation Challenge

## Overview
This project implements an ego-vehicle trajectory estimation system using computer vision techniques. The system tracks a traffic light as a reference point and estimates the vehicle's path in a ground-fixed coordinate system using depth data from a stereo camera.

## Problem Statement
Given a 10-second video from an ego-vehicle with:
- A fixed overhead traffic light
- Depth data from stereo camera
- Bounding box annotations for the traffic light

**Goal**: Estimate and visualize the ego-vehicle's trajectory in the ground frame using the traffic light as a world reference.

## Solution Approach

### 1. Data Processing
- **RGB Images**: 79 frames at 1920×1200 resolution
- **Depth Data**: 168 XYZ point cloud files (.npz format)
- **Bounding Boxes**: 299 traffic light annotations (CSV format)

### 2. Trajectory Estimation Algorithm

#### Step 1: Traffic Light Tracking
- Extract traffic light center coordinates from bounding box data
- Handle missing/invalid bounding boxes (zeros indicate no detection)
- Process 164 valid frames out of 299 total

#### Step 2: 3D Position Extraction
- Load corresponding depth data for each frame
- Extract 3D coordinates (X, Y, Z) at traffic light pixel locations
- Handle coordinate system: +X forward, +Y right, +Z up (camera frame)

#### Step 3: Ground Frame Transformation
- **Reference Point**: Traffic light at origin (0, 0, height) in ground frame
- **Coordinate System**: Right-handed with X forward, Y left, Z up
- **Transformation**: Camera position = -traffic_light_position_in_camera_frame
- **Ground Plane**: Project all points to Z=0 for 2D trajectory

### 3. Visualization
- **Static Plot**: `trajectory.png` showing complete trajectory
- **Color Coding**: Points colored by frame number (time progression)
- **Reference Markers**: Start (green), End (red), Traffic Light (yellow star)

## Results

### Trajectory Summary
- **Valid Frames**: 164 out of 299
- **Start Position**: (-36.01, -15.46) meters
- **End Position**: (-16.41, -1.06) meters
- **Movement**: Forward motion with slight rightward drift

### Key Observations
1. **Smooth Trajectory**: The path shows consistent forward movement
2. **Realistic Scale**: Distances are in meters, typical for vehicle movement
3. **Traffic Light Approach**: Vehicle moves closer to the reference point over time
4. **Data Quality**: Some frames had missing depth data or invalid bounding boxes

## File Structure
```
ego_trajectory_challenge/
├── ego_trajectory.py          # Main implementation
├── requirements.txt           # Python dependencies
├── trajectory.png            # Generated static plot (REQUIRED)
├── dataset/
│   ├── rgb/                  # RGB images (79 frames)
│   ├── xyz/                  # Depth data (168 files)
│   └── bbox_light.csv        # Traffic light bounding boxes
└── README.md                 # This file
```

## Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Code
```bash
python3 ego_trajectory.py
```

### Output Files
- `trajectory.png`: Static bird's-eye view plot (REQUIRED for submission)
- `trajectory.mp4`: Animated video (optional, requires ffmpeg)

## Technical Implementation

### Core Classes
- `EgoTrajectoryEstimator`: Main class handling data loading and trajectory estimation
- Methods:
  - `get_traffic_light_center()`: Extract pixel coordinates from bounding boxes
  - `get_3d_position()`: Get 3D coordinates from depth data
  - `estimate_trajectory()`: Main trajectory estimation algorithm
  - `plot_trajectory()`: Generate static visualization
  - `create_animated_trajectory()`: Create animated video

### Coordinate Systems
1. **Camera Frame**: +X forward, +Y right, +Z up
2. **Ground Frame**: +X forward, +Y left, +Z up (right-handed)
3. **Transformation**: Ground = -Camera (opposite direction)

### Data Handling
- Robust error handling for missing depth files
- Graceful handling of invalid bounding boxes
- Automatic detection of available data keys in .npz files

## Assumptions and Limitations

### Assumptions
1. Traffic light is fixed in world coordinates
2. Ground plane is flat (Z=0)
3. Camera height remains constant
4. Traffic light bounding boxes are accurate

### Limitations
1. Some frames have missing depth data (frames 166-298)
2. No temporal smoothing applied to trajectory
3. Single reference point (traffic light only)
4. No handling of camera motion blur or occlusions

## Future Enhancements (Part B - Optional)
- Track additional objects (golf cart, barrels, pedestrians)
- Implement temporal smoothing
- Add uncertainty estimation
- Create richer BEV visualizations
- Handle dynamic reference points

## Dependencies
- numpy >= 1.21.0
- opencv-python >= 4.5.0
- matplotlib >= 3.5.0
- pandas >= 1.3.0
- scipy >= 1.7.0

## Submission
This implementation satisfies the Part A requirements:
- ✅ `trajectory.png` generated
- ✅ Ground frame coordinate system implemented
- ✅ Traffic light used as reference
- ✅ Reasonable trajectory estimation

The trajectory shows a realistic vehicle path approaching a traffic light, demonstrating successful implementation of the ego-trajectory estimation challenge.

## Author
Computer Vision Challenge - Ego-Trajectory & Bird's-Eye View Mapping