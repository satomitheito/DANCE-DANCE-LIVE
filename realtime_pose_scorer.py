#!/usr/bin/env python3
"""
Real-time Pose Scoring Script
Compares live camera pose against recorded video landmarks and provides scoring
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import time
import os
import subprocess
import threading
from typing import Optional, Dict, List, Tuple
from scipy.spatial.distance import cosine
from scipy.optimize import linear_sum_assignment


class PoseScorer:
    """Real-time pose scoring against recorded video landmarks"""
    
    def __init__(self, landmarks_file: str, camera_index: int = 0):
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize pose detection
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Load recorded landmarks data
        self.recorded_landmarks = self.load_landmarks_data(landmarks_file)
        self.current_frame_index = 0
        self.total_recorded_frames = len(self.recorded_landmarks)
        
        # Load the original video for split screen
        self.original_video_path = landmarks_file.replace('_landmarks.json', '.mp4')
        self.video_cap = None
        if os.path.exists(self.original_video_path):
            self.video_cap = cv2.VideoCapture(self.original_video_path)
            print(f"Loaded original video: {self.original_video_path}")
        else:
            print(f"Warning: Original video not found at {self.original_video_path}")
        
        # Audio playback
        self.audio_process = None
        self.audio_started = False
        
        # Camera setup
        self.camera_index = camera_index
        self.cap = None
        
        # Scoring metrics
        self.scores_history = []
        self.current_score = 0.0
        self.avg_score = 0.0
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.total_frames = 0
        
        # Auto-advance settings
        self.auto_advance = True
        self.frame_advance_interval = 2  # Advance every 2 frames (0.07 seconds at 30fps) - Much faster!
        self.frames_since_advance = 0
        
        print(f"Loaded {self.total_recorded_frames} frames of recorded pose data")
    
    def load_landmarks_data(self, landmarks_file: str) -> List[np.ndarray]:
        """Load landmarks data from JSON file"""
        try:
            with open(landmarks_file, 'r') as f:
                data = json.load(f)
            
            landmarks_list = []
            for frame_data in data['frame_analysis']:
                landmarks = np.array(frame_data['landmarks'])
                landmarks_list.append(landmarks)
            
            return landmarks_list
        except Exception as e:
            print(f"Error loading landmarks data: {e}")
            return []
    
    def initialize_camera(self) -> bool:
        """Initialize camera capture"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                print(f"Error: Could not open camera {self.camera_index}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            print(f"Camera initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def calculate_pose_similarity(self, live_landmarks: np.ndarray, 
                                 recorded_landmarks: np.ndarray) -> float:
        """Calculate similarity between live and recorded poses with adaptive scaling"""
        if live_landmarks is None or recorded_landmarks is None:
            return 0.0
        
        # Normalize and scale landmarks to match live person's proportions
        live_normalized, recorded_normalized = self.normalize_and_scale_poses(live_landmarks, recorded_landmarks)
        
        # Calculate multiple similarity metrics
        similarities = []
        
        # 1. Cosine similarity (overall pose shape)
        live_flat = live_normalized.flatten()
        recorded_flat = recorded_normalized.flatten()
        cosine_sim = 1 - cosine(live_flat, recorded_flat)
        similarities.append(max(0, cosine_sim))
        
        # 2. Key point distance similarity
        key_points = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]  # Important joints
        key_distances = []
        for i in key_points:
            if i < len(live_normalized) and i < len(recorded_normalized):
                dist = np.linalg.norm(live_normalized[i] - recorded_normalized[i])
                key_distances.append(dist)
        
        if key_distances:
            avg_distance = np.mean(key_distances)
            distance_sim = max(0, 1 - (avg_distance / 0.3))  # Adjusted threshold for normalized poses
            similarities.append(distance_sim)
        
        # 3. Body part specific similarities
        body_part_sims = self.calculate_body_part_similarities(live_normalized, recorded_normalized)
        similarities.extend(body_part_sims)
        
        # 4. Pose orientation similarity
        orientation_sim = self.calculate_orientation_similarity(live_normalized, recorded_normalized)
        similarities.append(orientation_sim)
        
        # Weighted average of all similarities
        weights = [0.3, 0.25, 0.15, 0.15, 0.15]  # Adjusted weights
        final_score = np.average(similarities[:len(weights)], weights=weights[:len(similarities)])
        
        return min(1.0, max(0.0, final_score))
    
    def normalize_and_scale_poses(self, live_landmarks: np.ndarray, 
                                 recorded_landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize and scale poses to match live person's proportions"""
        if len(live_landmarks) == 0 or len(recorded_landmarks) == 0:
            return live_landmarks, recorded_landmarks
        
        # Calculate body scale factors
        live_scale = self.calculate_body_scale(live_landmarks)
        recorded_scale = self.calculate_body_scale(recorded_landmarks)
        
        # Scale recorded pose to match live person's size
        scale_factor = live_scale / recorded_scale if recorded_scale > 0 else 1.0
        
        # Normalize both poses to center
        live_centered = self.normalize_landmarks(live_landmarks)
        recorded_centered = self.normalize_landmarks(recorded_landmarks)
        
        # Scale recorded pose
        recorded_scaled = recorded_centered.copy()
        recorded_scaled[:, :2] *= scale_factor
        
        # Align poses by orientation
        live_aligned, recorded_aligned = self.align_pose_orientations(live_centered, recorded_scaled)
        
        return live_aligned, recorded_aligned
    
    def calculate_body_scale(self, landmarks: np.ndarray) -> float:
        """Calculate body scale based on shoulder-hip distance"""
        if len(landmarks) < 24:
            return 1.0
        
        # Use shoulder-hip distance as body scale reference
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        # Calculate shoulder width and hip width
        shoulder_width = np.linalg.norm(left_shoulder[:2] - right_shoulder[:2])
        hip_width = np.linalg.norm(left_hip[:2] - right_hip[:2])
        
        # Use average as body scale
        body_scale = (shoulder_width + hip_width) / 2
        
        return body_scale if body_scale > 0 else 1.0
    
    def align_pose_orientations(self, live_landmarks: np.ndarray, 
                               recorded_landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Align poses by rotating to match orientation"""
        if len(live_landmarks) < 12 or len(recorded_landmarks) < 12:
            return live_landmarks, recorded_landmarks
        
        # Use shoulder line for orientation alignment
        live_shoulder_vec = live_landmarks[12][:2] - live_landmarks[11][:2]
        recorded_shoulder_vec = recorded_landmarks[12][:2] - recorded_landmarks[11][:2]
        
        # Calculate rotation angle
        live_angle = np.arctan2(live_shoulder_vec[1], live_shoulder_vec[0])
        recorded_angle = np.arctan2(recorded_shoulder_vec[1], recorded_shoulder_vec[0])
        rotation_angle = live_angle - recorded_angle
        
        # Apply rotation to recorded landmarks
        cos_angle = np.cos(rotation_angle)
        sin_angle = np.sin(rotation_angle)
        rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
        
        recorded_aligned = recorded_landmarks.copy()
        recorded_aligned[:, :2] = recorded_landmarks[:, :2] @ rotation_matrix.T
        
        return live_landmarks, recorded_aligned
    
    def normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Normalize landmarks to center of mass"""
        if len(landmarks) == 0:
            return landmarks
        
        # Calculate center of mass
        center = np.mean(landmarks[:, :2], axis=0)
        
        # Normalize by subtracting center
        normalized = landmarks.copy()
        normalized[:, :2] -= center
        
        return normalized
    
    def calculate_body_part_similarities(self, live_landmarks: np.ndarray, 
                                       recorded_landmarks: np.ndarray) -> List[float]:
        """Calculate similarities for different body parts"""
        similarities = []
        
        # Define body part groups
        body_parts = {
            'arms': [11, 12, 13, 14, 15, 16],
            'legs': [23, 24, 25, 26, 27, 28],
            'torso': [11, 12, 23, 24, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        }
        
        for part_name, indices in body_parts.items():
            part_sim = 0.0
            valid_points = 0
            
            for idx in indices:
                if idx < len(live_landmarks) and idx < len(recorded_landmarks):
                    dist = np.linalg.norm(live_landmarks[idx] - recorded_landmarks[idx])
                    part_sim += max(0, 1 - (dist / 0.3))  # Normalize by expected distance
                    valid_points += 1
            
            if valid_points > 0:
                similarities.append(part_sim / valid_points)
        
        return similarities
    
    def calculate_orientation_similarity(self, live_landmarks: np.ndarray, 
                                       recorded_landmarks: np.ndarray) -> float:
        """Calculate orientation similarity between poses"""
        if len(live_landmarks) < 12 or len(recorded_landmarks) < 12:
            return 0.0
        
        # Calculate orientation vectors for different body parts
        orientations = []
        
        # Shoulder orientation
        live_shoulder_vec = live_landmarks[12][:2] - live_landmarks[11][:2]
        recorded_shoulder_vec = recorded_landmarks[12][:2] - recorded_landmarks[11][:2]
        orientations.append(self.vector_similarity(live_shoulder_vec, recorded_shoulder_vec))
        
        # Hip orientation
        live_hip_vec = live_landmarks[24][:2] - live_landmarks[23][:2]
        recorded_hip_vec = recorded_landmarks[24][:2] - recorded_landmarks[23][:2]
        orientations.append(self.vector_similarity(live_hip_vec, recorded_hip_vec))
        
        # Torso orientation (shoulder to hip)
        live_torso_vec = live_landmarks[24][:2] - live_landmarks[12][:2]
        recorded_torso_vec = recorded_landmarks[24][:2] - recorded_landmarks[12][:2]
        orientations.append(self.vector_similarity(live_torso_vec, recorded_torso_vec))
        
        return np.mean(orientations) if orientations else 0.0
    
    def vector_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate similarity between two vectors"""
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        
        # Normalize vectors
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)
        
        # Calculate cosine similarity
        similarity = np.dot(vec1_norm, vec2_norm)
        return max(0, similarity)
    
    def get_current_recorded_pose(self) -> Optional[np.ndarray]:
        """Get the current recorded pose based on frame index"""
        if self.current_frame_index < self.total_recorded_frames:
            return self.recorded_landmarks[self.current_frame_index]
        return None
    
    def get_current_video_frame(self) -> Optional[np.ndarray]:
        """Get the current video frame for split screen"""
        if self.video_cap is not None:
            # Set video to current frame
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)
            ret, frame = self.video_cap.read()
            if ret:
                return frame
        return None
    
    def start_audio(self):
        """Start audio playback using ffplay"""
        if self.original_video_path and os.path.exists(self.original_video_path) and not self.audio_started:
            try:
                # Use ffplay to play audio in background
                self.audio_process = subprocess.Popen([
                    'ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet',
                    self.original_video_path
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.audio_started = True
                print("🎵 Audio started!")
            except Exception as e:
                print(f"Could not start audio: {e}")
    
    def stop_audio(self):
        """Stop audio playback"""
        if self.audio_process:
            self.audio_process.terminate()
            self.audio_process = None
            self.audio_started = False
            print("🔇 Audio stopped!")
    
    def advance_frame(self):
        """Advance to next recorded frame"""
        self.current_frame_index = (self.current_frame_index + 1) % self.total_recorded_frames
    
    def calculate_fps(self) -> float:
        """Calculate current FPS"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
        
        return self.current_fps
    
    def draw_landmarks(self, frame: np.ndarray, landmarks: np.ndarray, 
                      color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """Draw pose landmarks on the frame"""
        annotated_frame = frame.copy()
        h, w, _ = frame.shape
        
        if landmarks is not None and len(landmarks) > 0:
            # Draw connections
            connections = self.mp_pose.POSE_CONNECTIONS
            for connection in connections:
                start_idx, end_idx = connection
                if (start_idx < len(landmarks) and end_idx < len(landmarks)):
                    start_point = (int(landmarks[start_idx][0] * w), int(landmarks[start_idx][1] * h))
                    end_point = (int(landmarks[end_idx][0] * w), int(landmarks[end_idx][1] * h))
                    
                    # Check if points are within frame bounds
                    if (0 <= start_point[0] < w and 0 <= start_point[1] < h and
                        0 <= end_point[0] < w and 0 <= end_point[1] < h):
                        cv2.line(annotated_frame, start_point, end_point, color, 2)
            
            # Draw landmarks
            for i, landmark in enumerate(landmarks):
                x = int(landmark[0] * w)
                y = int(landmark[1] * h)
                
                # Check if point is within frame bounds
                if 0 <= x < w and 0 <= y < h:
                    # Draw larger circles for better visibility
                    cv2.circle(annotated_frame, (x, y), 6, (255, 255, 255), -1)  # White background
                    cv2.circle(annotated_frame, (x, y), 6, color, 2)  # Colored outline
        
        return annotated_frame
    
    def draw_reference_landmarks(self, frame: np.ndarray, landmarks: np.ndarray, 
                                color: Tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
        """Draw reference landmarks simply and cleanly"""
        annotated_frame = frame.copy()
        h, w, _ = frame.shape
        
        if landmarks is not None and len(landmarks) > 0:
            # Draw connections
            connections = self.mp_pose.POSE_CONNECTIONS
            for connection in connections:
                start_idx, end_idx = connection
                if (start_idx < len(landmarks) and end_idx < len(landmarks)):
                    start_point = (int(landmarks[start_idx][0] * w), int(landmarks[start_idx][1] * h))
                    end_point = (int(landmarks[end_idx][0] * w), int(landmarks[end_idx][1] * h))
                    
                    # Check if points are within frame bounds
                    if (0 <= start_point[0] < w and 0 <= start_point[1] < h and
                        0 <= end_point[0] < w and 0 <= end_point[1] < h):
                        cv2.line(annotated_frame, start_point, end_point, color, 2)
            
            # Draw landmarks
            for i, landmark in enumerate(landmarks):
                x = int(landmark[0] * w)
                y = int(landmark[1] * h)
                
                # Check if point is within frame bounds
                if 0 <= x < w and 0 <= y < h:
                    # Simple hollow circles for reference
                    cv2.circle(annotated_frame, (x, y), 6, color, 2)  # Hollow circle
        
        return annotated_frame
    
    def draw_scoring_overlay(self, frame: np.ndarray, live_landmarks: Optional[np.ndarray], 
                           recorded_landmarks: Optional[np.ndarray], score: float) -> np.ndarray:
        """Draw scoring information overlay"""
        h, w = frame.shape[:2]
        
        # Background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Title
        cv2.putText(frame, "REAL-TIME POSE SCORING", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Current score
        score_color = (0, 255, 0) if score > 0.7 else (0, 255, 255) if score > 0.4 else (0, 0, 255)
        score_text = f"Score: {score:.1%}"
        cv2.putText(frame, score_text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2)
        
        # Average score
        avg_text = f"Average: {self.avg_score:.1%}"
        cv2.putText(frame, avg_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Frame info
        frame_text = f"Frame: {self.current_frame_index}/{self.total_recorded_frames}"
        cv2.putText(frame, frame_text, (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Auto-advance status
        auto_status = "AUTO" if self.auto_advance else "MANUAL"
        auto_color = (0, 255, 0) if self.auto_advance else (0, 255, 255)
        cv2.putText(frame, f"Mode: {auto_status}", (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, auto_color, 1)
        
        # FPS
        fps_text = f"FPS: {self.current_fps:.1f}"
        cv2.putText(frame, fps_text, (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Status
        if live_landmarks is not None and recorded_landmarks is not None:
            status_text = "SCORING ACTIVE - ADAPTIVE SCALING"
            status_color = (0, 255, 0)
        elif live_landmarks is not None:
            status_text = "WAITING FOR REFERENCE"
            status_color = (0, 255, 255)
        else:
            status_text = "NO POSE DETECTED"
            status_color = (0, 0, 255)
        
        cv2.putText(frame, status_text, (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        # Instructions
        instructions = "Green=You, Blue=Reference (scaled to your size)"
        cv2.putText(frame, instructions, (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Score bar
        bar_width = 200
        bar_height = 20
        bar_x = 20
        bar_y = 180
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        # Score bar
        score_width = int(bar_width * score)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + score_width, bar_y + bar_height), score_color, -1)
        
        return frame
    
    def run(self):
        """Main loop for real-time pose scoring"""
        if not self.initialize_camera():
            return
        
        print("\n🎯 Real-time Pose Scoring Started!")
        print("Press 'q' to quit, 'r' to reset scores, 'n' for next frame, 'p' for previous frame")
        print("Press 'a' to toggle auto-advance, 's' to slow down, 'f' to speed up")
        print("Press 'm' to toggle audio, 'space' to start/stop video")
        print("Left = You (Live), Right = Reference (Video)")
        
        # Start audio automatically
        self.start_audio()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Could not read from camera")
                    break
                
                # Flip camera horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                self.total_frames += 1
                self.frames_since_advance += 1
                
                # Calculate FPS
                self.calculate_fps()
                
                # Auto-advance to next frame if enabled
                if self.auto_advance and self.frames_since_advance >= self.frame_advance_interval:
                    self.advance_frame()
                    self.frames_since_advance = 0
                    # Debug: Print frame advancement
                    if self.total_frames % 60 == 0:  # Every 60 frames
                        print(f"Auto-advanced to frame {self.current_frame_index}")
                
                # Get current recorded pose and video frame
                recorded_landmarks = self.get_current_recorded_pose()
                video_frame = self.get_current_video_frame()
                
                # Process live pose detection
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb_frame)
                
                live_landmarks = None
                if results.pose_landmarks:
                    live_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
                
                # Calculate similarity score
                score = 0.0
                if live_landmarks is not None and recorded_landmarks is not None:
                    score = self.calculate_pose_similarity(live_landmarks, recorded_landmarks)
                    self.current_score = score
                    
                    # Update average score
                    self.scores_history.append(score)
                    if len(self.scores_history) > 100:  # Keep last 100 scores
                        self.scores_history.pop(0)
                    self.avg_score = np.mean(self.scores_history)
                
                # Draw landmarks on live camera
                if live_landmarks is not None:
                    frame = self.draw_landmarks(frame, live_landmarks, (0, 255, 0))  # Green for live
                
                # Create split screen
                h, w = frame.shape[:2]
                
                # Resize both frames to half width
                live_resized = cv2.resize(frame, (w//2, h))
                video_resized = None
                
                if video_frame is not None:
                    video_resized = cv2.resize(video_frame, (w//2, h))
                else:
                    # Create black frame if no video
                    video_resized = np.zeros((h, w//2, 3), dtype=np.uint8)
                    cv2.putText(video_resized, "No Video", (w//4 - 50, h//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Combine frames side by side
                split_frame = np.hstack((live_resized, video_resized))
                
                # Add labels
                cv2.putText(split_frame, "YOU (Live)", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(split_frame, "REFERENCE (Video)", (w//2 + 10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                # Draw scoring overlay on split frame
                split_frame = self.draw_scoring_overlay(split_frame, live_landmarks, recorded_landmarks, score)
                
                # Display split frame
                cv2.imshow('Real-time Pose Scoring - Split Screen', split_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Reset scores
                    self.scores_history = []
                    self.avg_score = 0.0
                    print("Scores reset!")
                elif key == ord('n'):
                    # Next frame
                    self.advance_frame()
                    print(f"Advanced to frame {self.current_frame_index}")
                elif key == ord('p'):
                    # Previous frame
                    self.current_frame_index = (self.current_frame_index - 1) % self.total_recorded_frames
                    print(f"Back to frame {self.current_frame_index}")
                elif key == ord('a'):
                    # Toggle auto-advance
                    self.auto_advance = not self.auto_advance
                    status = "ON" if self.auto_advance else "OFF"
                    print(f"Auto-advance {status}")
                elif key == ord('s'):
                    # Slow down (increase interval)
                    self.frame_advance_interval = min(60, self.frame_advance_interval + 5)
                    print(f"Slower: advancing every {self.frame_advance_interval} frames")
                elif key == ord('f'):
                    # Speed up (decrease interval)
                    self.frame_advance_interval = max(1, self.frame_advance_interval - 1)
                    print(f"Faster: advancing every {self.frame_advance_interval} frames")
                elif key == ord('m'):
                    # Toggle audio
                    if self.audio_started:
                        self.stop_audio()
                    else:
                        self.start_audio()
                elif key == ord(' '):  # Spacebar
                    # Toggle auto-advance
                    self.auto_advance = not self.auto_advance
                    status = "ON" if self.auto_advance else "OFF"
                    print(f"Video playback {status}")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_audio()
        if self.cap:
            self.cap.release()
        if self.video_cap:
            self.video_cap.release()
        cv2.destroyAllWindows()
        
        # Print final stats
        if self.scores_history:
            print(f"\n📊 Final Scoring Stats:")
            print(f"Average score: {self.avg_score:.1%}")
            print(f"Best score: {max(self.scores_history):.1%}")
            print(f"Total scores recorded: {len(self.scores_history)}")


def main():
    """Main function to run real-time pose scoring"""
    print("🎭 Real-time Pose Scoring")
    print("=" * 50)
    
    # Look for landmarks file
    landmarks_files = [
        "analyzed_pose_video_landmarks.json",
        "downloads/test_landmarks.json"
    ]
    
    landmarks_file = None
    for file_path in landmarks_files:
        if os.path.exists(file_path):
            landmarks_file = file_path
            break
    
    if not landmarks_file:
        print("❌ No landmarks file found!")
        print("Please run the full_body_analysis.py script first to generate landmarks data.")
        return
    
    print(f"Using landmarks file: {landmarks_file}")
    
    # Initialize and run scorer
    scorer = PoseScorer(landmarks_file)
    scorer.run()


if __name__ == "__main__":
    main()
