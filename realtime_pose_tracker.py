#!/usr/bin/env python3
"""
Real-time Pose Tracking Script using MediaPipe
Uses camera to track pose landmarks in real-time with live annotations
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from typing import Optional, Tuple


class RealtimePoseTracker:
    """Real-time pose tracking using camera and MediaPipe"""
    
    def __init__(self, camera_index: int = 0):
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize pose detection
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Faster for real-time
            enable_segmentation=False,  # Disable for better performance
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize camera
        self.camera_index = camera_index
        self.cap = None
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Pose tracking stats
        self.pose_detected_count = 0
        self.total_frames = 0
        
    def initialize_camera(self) -> bool:
        """Initialize camera capture"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                print(f"Error: Could not open camera {self.camera_index}")
                return False
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            print(f"Camera initialized successfully")
            print(f"Resolution: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            print(f"FPS: {int(self.cap.get(cv2.CAP_PROP_FPS))}")
            return True
            
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def draw_landmarks(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Draw pose landmarks on the frame with custom styling"""
        annotated_frame = frame.copy()
        h, w, _ = frame.shape
        
        # Define landmark connections for drawing
        connections = self.mp_pose.POSE_CONNECTIONS
        
        # Draw connections first (so they appear behind landmarks)
        for connection in connections:
            start_idx, end_idx = connection
            if (start_idx < len(landmarks) and end_idx < len(landmarks)):
                # Convert normalized coordinates to pixel coordinates
                start_point = (int(landmarks[start_idx][0] * w), int(landmarks[start_idx][1] * h))
                end_point = (int(landmarks[end_idx][0] * w), int(landmarks[end_idx][1] * h))
                
                # Draw line
                cv2.line(annotated_frame, start_point, end_point, (0, 255, 0), 2)
        
        # Draw landmarks as circles
        for i, landmark in enumerate(landmarks):
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
            cv2.circle(annotated_frame, (x, y), 6, (255, 255, 255), -1)  # White background
            cv2.circle(annotated_frame, (x, y), 6, color, 2)  # Colored outline
            
            # Add landmark number (smaller for real-time)
            cv2.putText(annotated_frame, str(i), (x + 8, y - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return annotated_frame
    
    def calculate_fps(self) -> float:
        """Calculate current FPS"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:  # Update every second
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
        
        return self.current_fps
    
    def analyze_pose_quality(self, landmarks: np.ndarray) -> dict:
        """Analyze pose quality in real-time"""
        if landmarks is None or len(landmarks) == 0:
            return {}
        
        # Calculate basic metrics
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        # Shoulder symmetry
        shoulder_symmetry = 1 - abs(left_shoulder[1] - right_shoulder[1])
        
        # Hip symmetry
        hip_symmetry = 1 - abs(left_hip[1] - right_hip[1])
        
        # Visibility score
        visibility_scores = landmarks[:, 2]
        avg_visibility = np.mean(visibility_scores)
        
        return {
            'shoulder_symmetry': max(0, min(1, shoulder_symmetry)),
            'hip_symmetry': max(0, min(1, hip_symmetry)),
            'visibility': avg_visibility
        }
    
    def draw_info_overlay(self, frame: np.ndarray, landmarks: Optional[np.ndarray], 
                         pose_metrics: dict) -> np.ndarray:
        """Draw information overlay on the frame"""
        h, w = frame.shape[:2]
        
        # Background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # FPS
        fps_text = f"FPS: {self.current_fps:.1f}"
        cv2.putText(frame, fps_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Pose detection status
        if landmarks is not None:
            status_text = "POSE DETECTED"
            color = (0, 255, 0)
            self.pose_detected_count += 1
        else:
            status_text = "NO POSE DETECTED"
            color = (0, 0, 255)
        
        cv2.putText(frame, status_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Pose metrics
        if pose_metrics:
            metrics_text = f"Visibility: {pose_metrics.get('visibility', 0):.2f}"
            cv2.putText(frame, metrics_text, (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            symmetry_text = f"Symmetry: {(pose_metrics.get('shoulder_symmetry', 0) + pose_metrics.get('hip_symmetry', 0))/2:.2f}"
            cv2.putText(frame, symmetry_text, (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Detection rate
        if self.total_frames > 0:
            detection_rate = (self.pose_detected_count / self.total_frames) * 100
            rate_text = f"Detection Rate: {detection_rate:.1f}%"
            cv2.putText(frame, rate_text, (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """Main loop for real-time pose tracking"""
        if not self.initialize_camera():
            return
        
        print("\n🎯 Real-time Pose Tracking Started!")
        print("Press 'q' to quit, 'r' to reset stats, 's' to save screenshot")
        print("Make sure you're visible in the camera frame for pose detection")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Could not read from camera")
                    break
                
                # Flip camera horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                self.total_frames += 1
                
                # Calculate FPS
                self.calculate_fps()
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process pose detection
                results = self.pose.process(rgb_frame)
                
                landmarks = None
                pose_metrics = {}
                
                if results.pose_landmarks:
                    # Extract landmarks
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
                    
                    # Analyze pose quality
                    pose_metrics = self.analyze_pose_quality(landmarks)
                    
                    # Draw landmarks
                    frame = self.draw_landmarks(frame, landmarks)
                
                # Draw info overlay
                frame = self.draw_info_overlay(frame, landmarks, pose_metrics)
                
                # Display frame
                cv2.imshow('Real-time Pose Tracking', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Reset stats
                    self.pose_detected_count = 0
                    self.total_frames = 0
                    print("Stats reset!")
                elif key == ord('s'):
                    # Save screenshot
                    screenshot_path = f"pose_screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(screenshot_path, frame)
                    print(f"Screenshot saved: {screenshot_path}")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Print final stats
        if self.total_frames > 0:
            detection_rate = (self.pose_detected_count / self.total_frames) * 100
            print(f"\n📊 Final Stats:")
            print(f"Total frames processed: {self.total_frames}")
            print(f"Pose detection rate: {detection_rate:.1f}%")
            print(f"Average FPS: {self.current_fps:.1f}")


def main():
    """Main function to run real-time pose tracking"""
    print("🎭 Real-time Pose Tracking with MediaPipe")
    print("=" * 50)
    
    # Try different camera indices if needed
    camera_indices = [0, 1, 2]  # Common camera indices
    
    for camera_index in camera_indices:
        print(f"\nTrying camera index {camera_index}...")
        tracker = RealtimePoseTracker(camera_index)
        
        if tracker.initialize_camera():
            print(f"✅ Camera {camera_index} initialized successfully!")
            tracker.run()
            break
        else:
            print(f"❌ Camera {camera_index} failed to initialize")
            if camera_index == camera_indices[-1]:
                print("\n❌ No cameras found! Please check your camera connection.")
                return
    else:
        print("\n❌ No working cameras found!")


if __name__ == "__main__":
    main()
