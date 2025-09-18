I used genAI to help me with this. I was going into it, I knew very little about how to track this challenge. I knew a little about cv2 and I know Python, but a lot of these concepts were new to me. I had AI write the code and I asked it to explain the concepts behind it for me to learn it.


Solution Approach

 1. Data Processing
- **RGB Images**: 79 frames at 1920×1200 resolution
- **Depth Data**: 168 XYZ point cloud files (.npz format)
- **Bounding Boxes**: 299 traffic light annotations (CSV format)

 2. Trajectory Estimation Algorithm

# Step 1: Traffic Light Tracking
- Extract traffic light center coordinates from bounding box data
- Handle missing/invalid bounding boxes (zeros indicate no detection)
- Process 164 valid frames out of 299 total

# Step 2: 3D Position Extraction
- Load corresponding depth data for each frame
- Extract 3D coordinates (X, Y, Z) at traffic light pixel locations
- Handle coordinate system: +X forward, +Y right, +Z up (camera frame)

# Step 3: Ground Frame Transformation
- Reference Point: traffic light at origin (0, 0, height) in ground frame
- **Coordinate System**: Right-handed with X forward, Y left, Z up
- **Transformation**: Camera position = -traffic_light_position_in_camera_frame
- **Ground Plane**: Project all points to Z=0 for 2D trajectory

 3. Visualization
- **Static Plot**: `trajectory.png` showing complete trajectory
- **Color Coding**: Points colored by frame number (time progression)
- **Reference Markers**: Start (green), End (red), Traffic Light (yellow star)

## Results

 Trajectory Summary
- **Valid Frames**: 164 out of 299
- **Start Position**: (-36.01, -15.46) meters
- **End Position**: (-16.41, -1.06) meters
- **Movement**: Forward motion with slight rightward drift

 Key Observations
1. **Smooth Trajectory**: The path shows consistent forward movement
2. **Realistic Scale**: Distances are in meters, typical for vehicle movement
3. **Traffic Light Approach**: Vehicle moves closer to the reference point over time
4. **Data Quality**: Some frames had missing depth data or invalid bounding boxes

## Files Generated
- `trajectory.png` - Static bird's-eye view plot (REQUIRED for submission)
- `ego_trajectory.py` - Main implementation code
- `requirements.txt` - Python dependencies

## How to Run
```bash
pip install -r requirements.txt
python3 ego_trajectory.py
```

## What I Learned
- How to work with depth data and 3D point clouds
- Coordinate system transformations between camera and ground frames
- Traffic light tracking using bounding box data
- Creating bird's-eye view visualizations
- The importance of handling missing data gracefully

This was a great learning experience that combined computer vision, 3D geometry, and data visualization!