#!/usr/bin/env python3
"""
Full Body Analysis Script using MediaPipe
Analyzes video for pose landmarks and creates annotated video with markers
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import subprocess
import tempfile
from typing import List, Tuple, Dict, Optional
import json
from datetime import datetime


class FullBodyAnalyzer:
    """Main class for full body analysis using MediaPipe"""
    
    def __init__(self):
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
        
        # Data storage
        self.landmarks_data = []
        self.frame_data = []
    
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
                      confidence_threshold: float = 0.1) -> np.ndarray:
        """Draw pose landmarks on the frame with custom styling"""
        annotated_frame = frame.copy()
        
        # Get frame dimensions
        h, w, _ = frame.shape
        
        # Define landmark connections for drawing
        connections = self.mp_pose.POSE_CONNECTIONS
        
        # Draw connections first (so they appear behind landmarks)
        for connection in connections:
            start_idx, end_idx = connection
            if (start_idx < len(landmarks) and end_idx < len(landmarks) and
                landmarks[start_idx][2] > confidence_threshold and 
                landmarks[end_idx][2] > confidence_threshold):
                
                # Convert normalized coordinates to pixel coordinates
                start_point = (int(landmarks[start_idx][0] * w), int(landmarks[start_idx][1] * h))
                end_point = (int(landmarks[end_idx][0] * w), int(landmarks[end_idx][1] * h))
                
                # Draw line
                cv2.line(annotated_frame, start_point, end_point, (0, 255, 0), 3)
        
        # Draw landmarks as circles
        for i, landmark in enumerate(landmarks):
            if landmark[2] > confidence_threshold:  # Check visibility
                x = int(landmark[0] * w)
                y = int(landmark[1] * h)
                
                # Different colors for different body parts
                if i in [11, 12, 13, 14, 15, 16]:  # Arms
                    color = (0, 0, 255)  # Red (BGR format)
                elif i in [23, 24, 25, 26, 27, 28]:  # Legs
                    color = (255, 0, 0)  # Blue (BGR format)
                elif i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:  # Face and shoulders
                    color = (0, 255, 255)  # Yellow (BGR format)
                else:  # Torso
                    color = (255, 0, 255)  # Magenta (BGR format)
                
                # Draw larger circle with outline
                cv2.circle(annotated_frame, (x, y), 8, (255, 255, 255), -1)  # White background
                cv2.circle(annotated_frame, (x, y), 8, color, 2)  # Colored outline
                
                # Add landmark number
                cv2.putText(annotated_frame, str(i), (x + 12, y - 12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        
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
                    annotated_frame = self.draw_landmarks(frame, landmarks, confidence_threshold=-1.0)
                    
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
                    analysis_results['frame_analysis'].append(frame_analysis)
                    
                else:
                    # No landmarks detected, use original frame
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
    
    def check_ffmpeg_available(self) -> bool:
        """Check if ffmpeg is available on the system"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def add_audio_to_video(self, video_path: str, audio_source_path: str, output_path: str) -> bool:
        """Add audio from source video to the processed video using ffmpeg"""
        if not self.check_ffmpeg_available():
            print("ffmpeg not found. Please install ffmpeg to add audio to videos.")
            print("Install with: brew install ffmpeg (on macOS) or apt install ffmpeg (on Ubuntu)")
            return False
            
        try:
            # Create a temporary file for the final output
            temp_output = output_path.replace('.mp4', '_temp.mp4')
            
            # Use ffmpeg to combine video and audio
            cmd = [
                'ffmpeg',
                '-i', video_path,  # Input video (no audio)
                '-i', audio_source_path,  # Input audio source
                '-c:v', 'copy',  # Copy video stream without re-encoding
                '-c:a', 'aac',  # Encode audio as AAC
                '-map', '0:v:0',  # Use video from first input
                '-map', '1:a:0',  # Use audio from second input
                '-shortest',  # End when shortest stream ends
                '-y',  # Overwrite output file
                temp_output
            ]
            
            print("Adding audio to video...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Replace the original file with the audio-enhanced version
                os.replace(temp_output, output_path)
                print(f"Audio successfully added to: {output_path}")
                return True
            else:
                print(f"Error adding audio: {result.stderr}")
                # Clean up temp file if it exists
                if os.path.exists(temp_output):
                    os.remove(temp_output)
                return False
                
        except Exception as e:
            print(f"Error adding audio: {e}")
            return False
    


def main():
    """Main function to run the full body analysis"""
    # Initialize analyzer
    analyzer = FullBodyAnalyzer()
    
    # Input and output paths
    input_video = "/Users/satomi/Documents/GitHub/DANCE-DANCE-LIVE/downloads/test.mp4"
    output_video = "/Users/satomi/Documents/GitHub/DANCE-DANCE-LIVE/analyzed_pose_video.mp4"
    
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
        
        # Add audio to the annotated video
        print("\nAdding audio to annotated video...")
        audio_added = analyzer.add_audio_to_video(output_video, input_video, output_video)
        
        if audio_added:
            print("✅ Audio successfully added to annotated video!")
        else:
            print("⚠️  Could not add audio (ffmpeg may not be installed)")
        
        print(f"\nAnalysis complete! Check the following files:")
        print(f"- Annotated video: {output_video}")
        print(f"- Landmarks data: {output_video.replace('.mp4', '_landmarks.json')}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
