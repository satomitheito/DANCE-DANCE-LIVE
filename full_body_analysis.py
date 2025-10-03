#!/usr/bin/env python3
"""
Full Body Analysis Script using PyTorch and MediaPipe
Analyzes video for pose landmarks and creates visualizations with markers
"""

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
from typing import List, Tuple, Dict, Optional
import json
from datetime import datetime


class PoseLandmarkDataset(Dataset):
    """PyTorch Dataset for pose landmark data"""
    
    def __init__(self, landmarks_data: List[np.ndarray], labels: Optional[List[int]] = None):
        self.landmarks_data = landmarks_data
        self.labels = labels if labels is not None else [0] * len(landmarks_data)
    
    def __len__(self):
        return len(self.landmarks_data)
    
    def __getitem__(self, idx):
        landmarks = torch.FloatTensor(self.landmarks_data[idx].flatten())
        label = torch.LongTensor([self.labels[idx]])
        return landmarks, label


class PoseAnalyzer(nn.Module):
    """PyTorch neural network for pose analysis"""
    
    def __init__(self, input_size: int = 99, hidden_size: int = 128, num_classes: int = 10):
        super(PoseAnalyzer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # Define the network architecture
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return self.softmax(x)


class FullBodyAnalyzer:
    """Main class for full body analysis using MediaPipe and PyTorch"""
    
    def __init__(self, model_path: Optional[str] = None):
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize pose detection
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize PyTorch model
        self.model = PoseAnalyzer()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        # Data storage
        self.landmarks_data = []
        self.frame_data = []
        
    def load_model(self, model_path: str):
        """Load a pre-trained PyTorch model"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def save_model(self, model_path: str):
        """Save the current model state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_size': self.model.input_size,
            'hidden_size': self.model.hidden_size,
            'num_classes': self.model.num_classes
        }, model_path)
        print(f"Model saved to {model_path}")
    
    def extract_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract pose landmarks from a single frame"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Extract landmark coordinates (33 landmarks, 3 coordinates each = 99 values)
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
                return landmarks
            return None
        except Exception as e:
            print(f"Error extracting landmarks: {e}")
            return None
    
    def draw_landmarks(self, frame: np.ndarray, landmarks: np.ndarray, 
                      confidence_threshold: float = 0.5) -> np.ndarray:
        """Draw pose landmarks on the frame with custom styling"""
        annotated_frame = frame.copy()
        
        # Define landmark connections for drawing
        connections = self.mp_pose.POSE_CONNECTIONS
        
        # Draw connections
        for connection in connections:
            start_idx, end_idx = connection
            if (start_idx < len(landmarks) and end_idx < len(landmarks) and
                landmarks[start_idx][2] > confidence_threshold and 
                landmarks[end_idx][2] > confidence_threshold):
                
                # Get frame dimensions
                h, w, _ = frame.shape
                
                # Convert normalized coordinates to pixel coordinates
                start_point = (int(landmarks[start_idx][0] * w), int(landmarks[start_idx][1] * h))
                end_point = (int(landmarks[end_idx][0] * w), int(landmarks[end_idx][1] * h))
                
                # Draw line
                cv2.line(annotated_frame, start_point, end_point, (0, 255, 0), 2)
        
        # Draw landmarks as circles
        for i, landmark in enumerate(landmarks):
            if landmark[2] > confidence_threshold:  # Check visibility
                h, w, _ = frame.shape
                x = int(landmark[0] * w)
                y = int(landmark[1] * h)
                
                # Different colors for different body parts
                if i in [11, 12, 13, 14, 15, 16]:  # Arms
                    color = (255, 0, 0)  # Red
                elif i in [23, 24, 25, 26, 27, 28]:  # Legs
                    color = (0, 0, 255)  # Blue
                elif i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:  # Face and shoulders
                    color = (255, 255, 0)  # Yellow
                else:  # Torso
                    color = (255, 0, 255)  # Magenta
                
                # Draw circle
                cv2.circle(annotated_frame, (x, y), 5, color, -1)
                
                # Add landmark number
                cv2.putText(annotated_frame, str(i), (x + 10, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        return annotated_frame
    
    def analyze_pose_quality(self, landmarks: np.ndarray) -> Dict[str, float]:
        """Analyze the quality and characteristics of the pose"""
        if landmarks is None or len(landmarks) == 0:
            return {}
        
        # Calculate pose quality metrics
        metrics = {}
        
        # Calculate body symmetry (left vs right)
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        # Shoulder symmetry
        shoulder_symmetry = 1 - abs(left_shoulder[1] - right_shoulder[1])
        metrics['shoulder_symmetry'] = max(0, min(1, shoulder_symmetry))
        
        # Hip symmetry
        hip_symmetry = 1 - abs(left_hip[1] - right_hip[1])
        metrics['hip_symmetry'] = max(0, min(1, hip_symmetry))
        
        # Calculate body proportions
        shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
        hip_width = abs(left_hip[0] - right_hip[0])
        metrics['shoulder_hip_ratio'] = shoulder_width / (hip_width + 1e-6)
        
        # Calculate pose stability (variance in landmark positions)
        landmark_variance = np.var(landmarks[:, :2], axis=0)
        metrics['pose_stability'] = 1 / (1 + np.mean(landmark_variance))
        
        # Calculate visibility score
        visibility_scores = landmarks[:, 2]
        metrics['visibility_score'] = np.mean(visibility_scores)
        
        return metrics
    
    def process_video(self, input_path: str, output_path: str, 
                     save_landmarks: bool = True) -> Dict[str, any]:
        """Process entire video and create analysis with markers"""
        print(f"Processing video: {input_path}")
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        analysis_results = {
            'video_info': {
                'width': width,
                'height': height,
                'fps': fps,
                'total_frames': total_frames
            },
            'frame_analysis': [],
            'overall_metrics': {}
        }
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extract landmarks
                landmarks = self.extract_landmarks(frame)
                
                if landmarks is not None:
                    # Store landmarks data
                    self.landmarks_data.append(landmarks)
                    
                    # Analyze pose quality
                    pose_metrics = self.analyze_pose_quality(landmarks)
                    
                    # Draw landmarks on frame
                    annotated_frame = self.draw_landmarks(frame, landmarks)
                    
                    # Add analysis info to frame
                    info_text = f"Frame: {frame_count}/{total_frames}"
                    cv2.putText(annotated_frame, info_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    if pose_metrics:
                        metrics_text = f"Visibility: {pose_metrics.get('visibility_score', 0):.2f}"
                        cv2.putText(annotated_frame, metrics_text, (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Store frame analysis
                    frame_analysis = {
                        'frame_number': frame_count,
                        'landmarks': landmarks.tolist(),
                        'metrics': pose_metrics,
                        'timestamp': frame_count / fps
                    }
                    print("landmarks detected")
                    analysis_results['frame_analysis'].append(frame_analysis)
                    
                else:
                    # No landmarks detected, use original frame
                    print("No landmarks detected")
                    annotated_frame = frame.copy()
                    cv2.putText(annotated_frame, "No pose detected", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Write frame to output video
                out.write(annotated_frame)
                frame_count += 1
                
                # Progress update
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        finally:
            cap.release()
            out.release()
        
        # Calculate overall metrics
        if self.landmarks_data:
            all_landmarks = np.array(self.landmarks_data)
            analysis_results['overall_metrics'] = {
                'total_frames_with_pose': len(self.landmarks_data),
                'pose_detection_rate': len(self.landmarks_data) / total_frames,
                'average_visibility': np.mean([np.mean(landmarks[:, 2]) for landmarks in self.landmarks_data])
            }
        
        # Save landmarks data if requested
        if save_landmarks and self.landmarks_data:
            landmarks_file = output_path.replace('.mp4', '_landmarks.json')
            with open(landmarks_file, 'w') as f:
                json.dump(analysis_results, f, indent=2)
            print(f"Landmarks data saved to: {landmarks_file}")
        
        print(f"Video processing complete. Output saved to: {output_path}")
        return analysis_results
    
    def create_pose_visualization(self, landmarks: np.ndarray, 
                                 save_path: Optional[str] = None) -> plt.Figure:
        """Create a 3D visualization of pose landmarks"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        if landmarks is not None:
            # Plot landmarks
            x = landmarks[:, 0]
            y = landmarks[:, 1]
            z = landmarks[:, 2]
            
            # Color by visibility
            colors = landmarks[:, 2]  # Use z-coordinate (visibility) as color
            
            scatter = ax.scatter(x, y, z, c=colors, cmap='viridis', s=50, alpha=0.7)
            
            # Add connections
            connections = self.mp_pose.POSE_CONNECTIONS
            for connection in connections:
                start_idx, end_idx = connection
                if (start_idx < len(landmarks) and end_idx < len(landmarks) and
                    landmarks[start_idx][2] > 0.5 and landmarks[end_idx][2] > 0.5):
                    
                    ax.plot([x[start_idx], x[end_idx]], 
                           [y[start_idx], y[end_idx]], 
                           [z[start_idx], z[end_idx]], 'b-', alpha=0.6)
            
            # Add landmark labels
            for i, (x_val, y_val, z_val) in enumerate(zip(x, y, z)):
                if z_val > 0.5:  # Only label visible landmarks
                    ax.text(x_val, y_val, z_val, str(i), fontsize=8)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z (Visibility)')
        ax.set_title('3D Pose Landmarks Visualization')
        
        plt.colorbar(scatter, label='Visibility Score')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D visualization saved to: {save_path}")
        
        return fig
    
    def train_pose_classifier(self, epochs: int = 100, learning_rate: float = 0.001):
        """Train a simple pose classifier using the collected landmarks data"""
        if len(self.landmarks_data) < 10:
            print("Not enough landmarks data for training. Need at least 10 samples.")
            return
        
        # Create dummy labels for demonstration (in real use, you'd have actual labels)
        labels = [i % 3 for i in range(len(self.landmarks_data))]  # 3 classes
        
        # Create dataset and dataloader
        dataset = PoseLandmarkDataset(self.landmarks_data, labels)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.model.train()
        
        print(f"Training pose classifier for {epochs} epochs...")
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_landmarks, batch_labels in dataloader:
                batch_landmarks = batch_landmarks.to(self.device)
                batch_labels = batch_labels.squeeze().to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_landmarks)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
        
        print("Training completed!")
        self.model.eval()


def main():
    """Main function to run the full body analysis"""
    # Initialize analyzer
    analyzer = FullBodyAnalyzer()
    
    # Input and output paths
    input_video = "/Users/satomi/Documents/GitHub/DANCE-DANCE-LIVE/downloads/test.mp4"
    output_video = "/Users/satomi/Documents/GitHub/DANCE-DANCE-LIVE/analyzed_pose_video.mp4"
    output_visualization = "/Users/satomi/Documents/GitHub/DANCE-DANCE-LIVE/pose_visualization.png"
    
    # Check if input video exists
    if not os.path.exists(input_video):
        print(f"Error: Input video not found at {input_video}")
        return
    
    try:
        # Process video
        print("Starting full body analysis...")
        results = analyzer.process_video(input_video, output_video, save_landmarks=True)
        
        # Print analysis summary
        print("\n=== ANALYSIS SUMMARY ===")
        print(f"Total frames processed: {results['video_info']['total_frames']}")
        print(f"Frames with pose detected: {results['overall_metrics']['total_frames_with_pose']}")
        print(f"Pose detection rate: {results['overall_metrics']['pose_detection_rate']:.2%}")
        print(f"Average visibility score: {results['overall_metrics']['average_visibility']:.3f}")
        
        # Create 3D visualization if we have landmarks
        if analyzer.landmarks_data:
            print("\nCreating 3D pose visualization...")
            fig = analyzer.create_pose_visualization(
                analyzer.landmarks_data[0],  # Use first frame's landmarks
                output_visualization
            )
            plt.show()
        
        # Train a simple classifier (optional)
        if len(analyzer.landmarks_data) > 10:
            print("\nTraining pose classifier...")
            analyzer.train_pose_classifier(epochs=50)
            
            # Save the trained model
            model_path = "/Users/satomi/Documents/GitHub/DANCE-DANCE-LIVE/pose_model.pth"
            analyzer.save_model(model_path)
        
        print(f"\nAnalysis complete! Check the following files:")
        print(f"- Annotated video: {output_video}")
        print(f"- 3D visualization: {output_visualization}")
        print(f"- Landmarks data: {output_video.replace('.mp4', '_landmarks.json')}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
