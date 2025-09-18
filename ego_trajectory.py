#!/usr/bin/env python3
"""
Ego-Vehicle Trajectory Estimation
Computer Vision Challenge: Ego-Trajectory & Bird's-Eye View Mapping
"""

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import os

class EgoTrajectoryEstimator:
    def __init__(self, dataset_path="dataset"):
        self.dataset_path = Path(dataset_path)
        self.rgb_path = self.dataset_path / "rgb"
        self.xyz_path = self.dataset_path / "xyz"
        self.bbox_file = self.dataset_path / "bbox_light.csv"
        
        # Load bounding box data
        self.bboxes = pd.read_csv(self.bbox_file)
        print(f"Loaded {len(self.bboxes)} bounding box entries")
        print(f"Bounding box columns: {self.bboxes.columns.tolist()}")
        print(f"First few entries:\n{self.bboxes.head()}")
        
        # Get available frames
        self.rgb_frames = sorted([f for f in os.listdir(self.rgb_path) if f.endswith('.png')])
        self.xyz_frames = sorted([f for f in os.listdir(self.xyz_path) if f.endswith('.npz')])
        
        print(f"Found {len(self.rgb_frames)} RGB frames")
        print(f"Found {len(self.xyz_frames)} XYZ frames")
        
    def get_traffic_light_center(self, frame_id):
        """Get the center pixel coordinates of the traffic light for a given frame"""
        if frame_id >= len(self.bboxes):
            return None, None
            
        bbox = self.bboxes.iloc[frame_id]
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        
        # Check if bounding box is valid (not all zeros)
        if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
            return None, None
            
        # Calculate center
        center_u = (x1 + x2) / 2
        center_v = (y1 + y2) / 2
        
        return int(center_u), int(center_v)
    
    def get_3d_position(self, frame_id, u, v):
        """Get 3D position from depth data for given pixel coordinates"""
        # Find corresponding XYZ file
        xyz_file = f"depth{frame_id:06d}.npz"
        xyz_path = self.xyz_path / xyz_file
        
        if not xyz_path.exists():
            print(f"XYZ file not found: {xyz_path}")
            return None
            
        try:
            # Load depth data
            xyz_data = np.load(xyz_path)
            # Check what keys are available
            if 'xyz' in xyz_data:
                points = xyz_data['xyz']
            elif 'points' in xyz_data:
                points = xyz_data['points']
            else:
                # Try to get the first array
                key = list(xyz_data.keys())[0]
                points = xyz_data[key]
            
            # Get 3D coordinates at pixel (u, v)
            if 0 <= v < points.shape[0] and 0 <= u < points.shape[1]:
                X, Y, Z = points[v, u, :3]  # Take only first 3 channels (X, Y, Z)
                return np.array([X, Y, Z])
            else:
                print(f"Pixel coordinates ({u}, {v}) out of bounds for frame {frame_id}")
                return None
                
        except Exception as e:
            print(f"Error loading XYZ data for frame {frame_id}: {e}")
            return None
    
    def estimate_trajectory(self):
        """Estimate ego-vehicle trajectory using traffic light as reference"""
        trajectory_points = []
        valid_frames = []
        
        print("\n=== Estimating Trajectory ===")
        
        # Process all frames
        for frame_id in range(len(self.bboxes)):
            u, v = self.get_traffic_light_center(frame_id)
            if u is not None and v is not None:
                # Get 3D position of traffic light in camera coordinates
                pos_3d = self.get_3d_position(frame_id, u, v)
                if pos_3d is not None:
                    trajectory_points.append(pos_3d)
                    valid_frames.append(frame_id)
                    print(f"Frame {frame_id}: Traffic light at ({pos_3d[0]:.2f}, {pos_3d[1]:.2f}, {pos_3d[2]:.2f})")
        
        if len(trajectory_points) < 2:
            print("Not enough valid frames for trajectory estimation")
            return None, None
            
        trajectory_points = np.array(trajectory_points)
        
        # Define ground frame coordinate system
        # Origin: directly under the traffic light on the ground
        # At t=0, line joining car and traffic light is aligned with +X axis
        # Right-handed system: X forward, Y left, Z up
        
        # Use the first valid frame as reference
        ref_pos = trajectory_points[0]  # Traffic light position in camera frame at t=0
        
        # Ground frame transformation
        # The traffic light is at (0, 0, height) in ground frame
        # Camera is at some position (x, y, 0) in ground frame
        # We need to find the camera's position relative to the traffic light
        
        ground_trajectory = []
        
        for i, cam_pos in enumerate(trajectory_points):
            # Camera position relative to traffic light
            # In camera frame: traffic light is at cam_pos
            # In ground frame: camera is at -cam_pos (opposite direction)
            ground_x = -cam_pos[0]  # Forward direction
            ground_y = -cam_pos[1]  # Left direction (camera Y is right, ground Y is left)
            ground_z = 0  # Ground plane
            
            ground_trajectory.append([ground_x, ground_y, ground_z])
        
        ground_trajectory = np.array(ground_trajectory)
        
        return ground_trajectory, valid_frames
    
    def plot_trajectory(self, trajectory, valid_frames, save_path="trajectory.png"):
        """Plot the ego-vehicle trajectory in BEV coordinates"""
        plt.figure(figsize=(12, 8))
        
        # Plot trajectory
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Ego-vehicle trajectory')
        plt.scatter(trajectory[:, 0], trajectory[:, 1], c=valid_frames, cmap='viridis', s=50, alpha=0.7)
        
        # Mark start and end points
        plt.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=100, marker='o', label='Start')
        plt.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=100, marker='s', label='End')
        
        # Mark traffic light position (origin)
        plt.scatter(0, 0, c='yellow', s=200, marker='*', label='Traffic Light (Reference)')
        
        plt.xlabel('X (meters) - Forward Direction')
        plt.ylabel('Y (meters) - Left Direction')
        plt.title('Ego-Vehicle Trajectory in Ground Frame\n(Traffic Light as Reference)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # Add some padding
        margin = 5
        plt.xlim(trajectory[:, 0].min() - margin, trajectory[:, 0].max() + margin)
        plt.ylim(trajectory[:, 1].min() - margin, trajectory[:, 1].max() + margin)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Trajectory plot saved to {save_path}")
    
    def create_animated_trajectory(self, trajectory, valid_frames, save_path="trajectory.mp4"):
        """Create animated trajectory video"""
        from matplotlib.animation import FuncAnimation
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Initialize plot elements
        line, = ax.plot([], [], 'b-', linewidth=2, label='Ego-vehicle trajectory')
        scatter = ax.scatter([], [], c=[], cmap='viridis', s=50, alpha=0.7)
        start_point = ax.scatter([], [], c='green', s=100, marker='o', label='Start')
        end_point = ax.scatter([], [], c='red', s=100, marker='s', label='End')
        ref_point = ax.scatter(0, 0, c='yellow', s=200, marker='*', label='Traffic Light (Reference)')
        
        ax.set_xlabel('X (meters) - Forward Direction')
        ax.set_ylabel('Y (meters) - Left Direction')
        ax.set_title('Ego-Vehicle Trajectory Animation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Set axis limits
        margin = 5
        ax.set_xlim(trajectory[:, 0].min() - margin, trajectory[:, 0].max() + margin)
        ax.set_ylim(trajectory[:, 1].min() - margin, trajectory[:, 1].max() + margin)
        
        def animate(frame_idx):
            if frame_idx < len(trajectory):
                # Update trajectory line
                line.set_data(trajectory[:frame_idx+1, 0], trajectory[:frame_idx+1, 1])
                
                # Update scatter points
                scatter.set_offsets(trajectory[:frame_idx+1, :2])
                scatter.set_array(valid_frames[:frame_idx+1])
                
                # Update start point
                if frame_idx == 0:
                    start_point.set_offsets([[trajectory[0, 0], trajectory[0, 1]]])
                
                # Update end point
                end_point.set_offsets([[trajectory[frame_idx, 0], trajectory[frame_idx, 1]]])
            
            return line, scatter, start_point, end_point
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(trajectory), 
                           interval=100, blit=False, repeat=True)
        
        # Save animation
        anim.save(save_path, writer='ffmpeg', fps=10)
        plt.close()
        
        print(f"Animated trajectory saved to {save_path}")

def main():
    print("=== Ego-Vehicle Trajectory Estimation ===")
    
    # Initialize estimator
    estimator = EgoTrajectoryEstimator()
    
    # Estimate trajectory
    trajectory, valid_frames = estimator.estimate_trajectory()
    
    if trajectory is not None:
        print(f"\n=== Trajectory Summary ===")
        print(f"Total valid frames: {len(valid_frames)}")
        print(f"Trajectory length: {len(trajectory)} points")
        print(f"Start position: ({trajectory[0, 0]:.2f}, {trajectory[0, 1]:.2f})")
        print(f"End position: ({trajectory[-1, 0]:.2f}, {trajectory[-1, 1]:.2f})")
        
        # Create static plot
        print("\n=== Creating Static Plot ===")
        estimator.plot_trajectory(trajectory, valid_frames)
        
        # Create animated video
        print("\n=== Creating Animated Video ===")
        try:
            estimator.create_animated_trajectory(trajectory, valid_frames)
        except Exception as e:
            print(f"Could not create animated video: {e}")
            print("This might be due to missing ffmpeg. Static plot is still available.")
        
        print("\n=== Challenge Complete! ===")
        print("Generated files:")
        print("- trajectory.png (static plot)")
        print("- trajectory.mp4 (animated video, if ffmpeg available)")
    else:
        print("Failed to estimate trajectory. Check data quality.")

if __name__ == "__main__":
    main()
