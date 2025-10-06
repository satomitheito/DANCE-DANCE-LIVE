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
        
        # Display settings
        self.fullscreen = False
        self.window_created = False
        self.optimal_width = 1920  # Default to 1920x1080
        self.optimal_height = 1080
        
        # Feedback smoothing
        self.feedback_history = []
        self.feedback_smoothing_window = 10  # Average over last 10 scores
        
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
        self.frame_advance_interval = 1.0/30.0  # 30 FPS (original video speed)
        self.frames_since_advance = 0
        self.last_advance_time = time.time()
        
        # Frame skipping settings
        self.frame_skip = 2  # Start with 2x speed (skip every 2nd frame)
        self.skip_increment = 1  # Increase/decrease by 1 for smoother changes
        
        # Get original video FPS
        if self.video_cap:
            self.original_fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            if self.original_fps > 0:
                self.frame_advance_interval = 1.0 / self.original_fps
                print(f"Original video FPS: {self.original_fps:.1f}")
            else:
                self.original_fps = 30.0
        else:
            self.original_fps = 30.0
        
        print(f"Loaded {self.total_recorded_frames} frames of recorded pose data")
    
    def get_optimal_screen_size(self):
        """Get optimal screen size for display"""
        # Use a large but reasonable size for most screens
        self.optimal_width = 1600
        self.optimal_height = 900
        print(f"Using optimized size: {self.optimal_width}x{self.optimal_height}")
    
    def get_user_feedback(self, score: float) -> Tuple[str, Tuple[int, int, int]]:
        """Get user-friendly feedback based on smoothed score"""
        # Add current score to history
        self.feedback_history.append(score)
        
        # Keep only recent scores for smoothing
        if len(self.feedback_history) > self.feedback_smoothing_window:
            self.feedback_history.pop(0)
        
        # Calculate smoothed score
        smoothed_score = np.mean(self.feedback_history)
        
        # Get feedback based on smoothed score
        if smoothed_score >= 0.8:  # 80% or higher
            return "PERFECT!", (0, 255, 0)  # Green
        elif smoothed_score >= 0.7:  # 70-79%
            return "EXCELLENT!", (0, 255, 100)  # Light green
        elif smoothed_score >= 0.6:  # 60-69%
            return "GREAT!", (100, 255, 0)  # Yellow-green
        elif smoothed_score >= 0.5:  # 50-59%
            return "GOOD!", (255, 255, 0)  # Yellow
        elif smoothed_score >= 0.4:  # 40-49%
            return "OKAY", (255, 165, 0)  # Orange
        elif smoothed_score >= 0.3:  # 30-39%
            return "KEEP TRYING", (255, 100, 0)  # Red-orange
        else:  # Below 30%
            return "FOCUS UP!", (255, 0, 0)  # Red
    
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
        
        # 1. Cosine similarity (nose + body pose shape, excluding other face landmarks)
        # Include nose (0) and body landmarks (11+), skip other face landmarks (1-10)
        if len(live_normalized) > 10 and len(recorded_normalized) > 10:
            # Create arrays with nose + body landmarks only
            live_filtered = np.vstack([live_normalized[0:1], live_normalized[11:]])  # Nose + body
            recorded_filtered = np.vstack([recorded_normalized[0:1], recorded_normalized[11:]])  # Nose + body
            live_flat = live_filtered.flatten()
            recorded_flat = recorded_filtered.flatten()
            cosine_sim = 1 - cosine(live_flat, recorded_flat)
            similarities.append(max(0, cosine_sim))
        else:
            # Fallback to full pose if not enough landmarks
            live_flat = live_normalized.flatten()
            recorded_flat = recorded_normalized.flatten()
            cosine_sim = 1 - cosine(live_flat, recorded_flat)
            similarities.append(max(0, cosine_sim))
        
        # 2. Key point distance similarity (nose + body joints only)
        # Include nose (0) and body joints (11+), skip other face landmarks (1-10)
        key_points = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]  # Nose + body joints
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
        
        # Define body part groups (excluding face landmarks)
        body_parts = {
            'arms': [11, 12, 13, 14, 15, 16],
            'legs': [23, 24, 25, 26, 27, 28],
            'torso': [11, 12, 23, 24]  # Only body joints, no face
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
        """Advance to next recorded frame with skipping"""
        # Skip frames based on frame_skip setting
        self.current_frame_index = self.current_frame_index + self.frame_skip
        # Don't loop - stop when we reach the end
        if self.current_frame_index >= self.total_recorded_frames:
            self.current_frame_index = self.total_recorded_frames - 1  # Stay at last frame
        self.last_advance_time = time.time()
    
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
            # Draw connections (skip face connections)
            connections = self.mp_pose.POSE_CONNECTIONS
            for connection in connections:
                start_idx, end_idx = connection
                # Skip connections involving face landmarks (1-10), but allow nose (0)
                if (start_idx > 0 and start_idx <= 10) or (end_idx > 0 and end_idx <= 10):
                    continue
                    
                if (start_idx < len(landmarks) and end_idx < len(landmarks)):
                    start_point = (int(landmarks[start_idx][0] * w), int(landmarks[start_idx][1] * h))
                    end_point = (int(landmarks[end_idx][0] * w), int(landmarks[end_idx][1] * h))
                    
                    # Check if points are within frame bounds
                    if (0 <= start_point[0] < w and 0 <= start_point[1] < h and
                        0 <= end_point[0] < w and 0 <= end_point[1] < h):
                        cv2.line(annotated_frame, start_point, end_point, color, 2)
            
            # Draw landmarks (nose + body landmarks only)
            for i, landmark in enumerate(landmarks):
                # Only draw nose (0) and body landmarks (11+)
                if i > 0 and i <= 10:
                    continue
                    
                x = int(landmark[0] * w)
                y = int(landmark[1] * h)
                
                # Check if point is within frame bounds
                if 0 <= x < w and 0 <= y < h:
                    if i == 0:  # Nose - small dot
                        cv2.circle(annotated_frame, (x, y), 3, color, -1)  # Small dot for nose
                    else:  # Body landmarks - larger circles
                        cv2.circle(annotated_frame, (x, y), 6, (255, 255, 255), -1)  # White background
                        cv2.circle(annotated_frame, (x, y), 6, color, 2)  # Colored outline
        
        return annotated_frame
    
    def draw_reference_landmarks(self, frame: np.ndarray, landmarks: np.ndarray, 
                                color: Tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
        """Draw reference landmarks simply and cleanly"""
        annotated_frame = frame.copy()
        h, w, _ = frame.shape
        
        if landmarks is not None and len(landmarks) > 0:
            # Draw connections (skip face connections)
            connections = self.mp_pose.POSE_CONNECTIONS
            for connection in connections:
                start_idx, end_idx = connection
                # Skip connections involving face landmarks (1-10), but allow nose (0)
                if (start_idx > 0 and start_idx <= 10) or (end_idx > 0 and end_idx <= 10):
                    continue
                    
                if (start_idx < len(landmarks) and end_idx < len(landmarks)):
                    start_point = (int(landmarks[start_idx][0] * w), int(landmarks[start_idx][1] * h))
                    end_point = (int(landmarks[end_idx][0] * w), int(landmarks[end_idx][1] * h))
                    
                    # Check if points are within frame bounds
                    if (0 <= start_point[0] < w and 0 <= start_point[1] < h and
                        0 <= end_point[0] < w and 0 <= end_point[1] < h):
                        cv2.line(annotated_frame, start_point, end_point, color, 2)
            
            # Draw landmarks (nose + body landmarks only)
            for i, landmark in enumerate(landmarks):
                # Only draw nose (0) and body landmarks (11+)
                if i > 0 and i <= 10:
                    continue
                    
                x = int(landmark[0] * w)
                y = int(landmark[1] * h)
                
                # Check if point is within frame bounds
                if 0 <= x < w and 0 <= y < h:
                    if i == 0:  # Nose - small dot
                        cv2.circle(annotated_frame, (x, y), 2, color, -1)  # Very small dot for nose
                    else:  # Body landmarks - hollow circles
                        cv2.circle(annotated_frame, (x, y), 6, color, 2)  # Hollow circle for body
        
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
        
        # Get optimal screen size
        self.get_optimal_screen_size()
        
        print("\n🎯 Real-time Pose Scoring Started!")
        print("Press 'q' to quit, 'r' to reset scores, 'n' for next frame, 'p' for previous frame")
        print("Press 'a' to toggle auto-advance, 's' to slow down, 'f' to speed up")
        print("Press 'm' to toggle audio, 'space' to start/stop video")
        print("Press '+' to increase frame skip, '-' to decrease frame skip")
        print("Press '1' for 1x speed, '2' for 2x speed, '3' for 3x speed, '4' for 5x speed")
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
                
                # Auto-advance to next frame at original video speed
                if self.auto_advance:
                    current_time = time.time()
                    # Advance based on the frame interval (time-based)
                    if current_time - self.last_advance_time >= self.frame_advance_interval:
                        old_frame = self.current_frame_index
                        self.advance_frame()
                        
                        # Check if video has ended
                        if self.current_frame_index >= self.total_recorded_frames - 1:
                            print(f"\n🎬 Video ended at frame {self.current_frame_index + 1}/{self.total_recorded_frames}")
                            print(f"📊 Final Results:")
                            print(f"Average accuracy: {self.avg_score:.1%}")
                            print(f"Best accuracy: {max(self.scores_history) if self.scores_history else 0:.1%}")
                            print(f"Total frames processed: {self.total_frames}")
                            print("\nPress 'q' to quit or 'r' to restart")
                            self.auto_advance = False  # Stop auto-advancing
                        
                        # Debug: Print frame advancement
                        if self.total_frames % 60 == 0:  # Every 60 frames
                            print(f"Auto-advanced to frame {self.current_frame_index + 1}/{self.total_recorded_frames}")
                
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
                
                # Create split screen with optimal sizing
                h, w = frame.shape[:2]
                
                # Calculate optimal dimensions
                target_height = self.optimal_height
                target_width = self.optimal_width
                
                # Calculate aspect ratio for each side (50/50 split)
                live_width = target_width // 2
                video_width = target_width // 2
                
                # Resize both frames to optimal dimensions
                live_resized = cv2.resize(frame, (live_width, target_height))
                video_resized = None
                
                if video_frame is not None:
                    video_resized = cv2.resize(video_frame, (video_width, target_height))
                else:
                    # Create black frame if no video
                    video_resized = np.zeros((target_height, video_width, 3), dtype=np.uint8)
                    cv2.putText(video_resized, "No Video", (video_width//2 - 50, target_height//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Combine frames side by side
                split_frame = np.hstack((live_resized, video_resized))
                
                # Scale text size based on screen size
                total_width = live_width + video_width
                scale_factor = max(1.0, total_width / 1200)  # Scale up for larger screens
                font_scale = 1.0 * scale_factor
                thickness = int(2 * scale_factor)
                
                # Add labels with larger text
                cv2.putText(split_frame, "YOU (Live)", (20, int(50 * scale_factor)), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
                cv2.putText(split_frame, "REFERENCE (Video)", (live_width + 20, int(50 * scale_factor)), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness)
                
                # Simple accuracy percentage in top right with larger text
                # User-friendly feedback system
                feedback_text, feedback_color = self.get_user_feedback(score)
                
                # Main feedback text (large and prominent)
                feedback_size = cv2.getTextSize(feedback_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.5, thickness * 2)[0]
                feedback_x = (total_width - feedback_size[0]) // 2
                feedback_y = int(80 * scale_factor)
                
                # Draw background rectangle for feedback
                cv2.rectangle(split_frame, (feedback_x - 20, feedback_y - int(50 * scale_factor)), 
                             (feedback_x + feedback_size[0] + 20, feedback_y + int(20 * scale_factor)), (0, 0, 0), -1)
                cv2.putText(split_frame, feedback_text, (feedback_x, feedback_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.5, feedback_color, thickness * 2)
                
                # Accuracy percentage (smaller, in top right)
                accuracy_text = f"{score:.0%}"
                acc_size = cv2.getTextSize(accuracy_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, thickness)[0]
                acc_x = total_width - acc_size[0] - 20
                acc_y = int(40 * scale_factor)
                cv2.putText(split_frame, accuracy_text, (acc_x, acc_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (200, 200, 200), thickness)
                
                # Add frame skip info in bottom right
                skip_text = f"Skip: {self.frame_skip}x"
                skip_text_size = cv2.getTextSize(skip_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, thickness)[0]
                skip_x = total_width - skip_text_size[0] - 20
                skip_y = target_height - 20
                cv2.putText(split_frame, skip_text, (skip_x, skip_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, (200, 200, 200), thickness)
                
                # Add video end indicator if video has ended
                if self.current_frame_index >= self.total_recorded_frames - 1:
                    end_text = "VIDEO ENDED - Press 'r' to restart or 'q' to quit"
                    end_text_size = cv2.getTextSize(end_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, thickness)[0]
                    end_x = (total_width - end_text_size[0]) // 2
                    end_y = target_height - 60
                    # Draw background rectangle
                    cv2.rectangle(split_frame, (end_x - 10, end_y - 30), (end_x + end_text_size[0] + 10, end_y + 10), (0, 0, 0), -1)
                    cv2.putText(split_frame, end_text, (end_x, end_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (0, 255, 255), thickness)
                
                # Display split frame in optimal size
                if not self.window_created:
                    cv2.namedWindow('Real-time Pose Scoring - Split Screen', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('Real-time Pose Scoring - Split Screen', total_width, target_height)
                    self.window_created = True
                cv2.imshow('Real-time Pose Scoring - Split Screen', split_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Reset scores and restart video
                    self.scores_history = []
                    self.avg_score = 0.0
                    self.current_frame_index = 0
                    self.auto_advance = True
                    print("Scores reset and video restarted!")
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
                    # Slow down video (increase time interval)
                    self.frame_advance_interval = min(2.0, self.frame_advance_interval + 0.05)
                    print(f"Slower: {1.0/self.frame_advance_interval:.1f} FPS")
                elif key == ord('f'):
                    # Speed up video (decrease time interval)
                    self.frame_advance_interval = max(0.016, self.frame_advance_interval - 0.05)
                    print(f"Faster: {1.0/self.frame_advance_interval:.1f} FPS")
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
                elif key == ord('+') or key == ord('='):  # Plus key
                    # Increase frame skip
                    self.frame_skip = min(8, self.frame_skip + self.skip_increment)
                    print(f"Frame skip: {self.frame_skip}x (skipping every {self.frame_skip} frames)")
                elif key == ord('-') or key == ord('_'):  # Minus key
                    # Decrease frame skip
                    self.frame_skip = max(1, self.frame_skip - self.skip_increment)
                    print(f"Frame skip: {self.frame_skip}x (skipping every {self.frame_skip} frames)")
                elif key == ord('1'):  # 1x speed
                    self.frame_skip = 1
                    print("Speed: 1x (normal speed)")
                elif key == ord('2'):  # 2x speed
                    self.frame_skip = 2
                    print("Speed: 2x (fast)")
                elif key == ord('3'):  # 3x speed
                    self.frame_skip = 3
                    print("Speed: 3x (faster)")
                elif key == ord('4'):  # 5x speed
                    self.frame_skip = 5
                    print("Speed: 5x (very fast)")
        
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
