#!/usr/bin/env python3
"""
Dance Dance Live - Flask Application
A Just Dance-style game using computer vision for pose matching
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import json
import os
import base64
from pathlib import Path

from pose_comparison import compare_poses


# Configuration
VIDEO_FILE = "analyzed_pose_video.mp4"
LANDMARKS_FILE = "analyzed_pose_video_landmarks.json"


app = Flask(__name__)
CORS(app)  # Enable CORS for development


class PoseAnalysisService:
    """Service for analyzing poses from webcam frames"""

    def __init__(self, landmarks_file):
        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Load reference landmarks
        self.reference_landmarks = self._load_landmarks(landmarks_file)
        self.video_info = self.reference_landmarks.get('video_info', {})

    def _load_landmarks(self, filepath):
        """Load landmarks from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Create frame lookup
        frame_lookup = {}
        for frame_data in data.get('frame_analysis', []):
            frame_num = frame_data['frame_number']
            landmarks = np.array(frame_data['landmarks'])
            frame_lookup[frame_num] = landmarks

        return {
            'frame_lookup': frame_lookup,
            'video_info': data.get('video_info', {}),
            'overall_metrics': data.get('overall_metrics', {})
        }

    def extract_landmarks(self, frame_data):
        """Extract pose landmarks from base64 encoded frame"""
        try:
            # Decode base64 image
            img_data = base64.b64decode(frame_data.split(',')[1])
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe
            results = self.pose_detector.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = np.array([
                    [lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark
                ])
                return landmarks
            return None
        except Exception as e:
            print(f"Error extracting landmarks: {e}")
            return None

    def score_pose(self, user_landmarks, frame_number):
        """Compare user pose to reference pose at given frame"""
        ref_landmarks = self.reference_landmarks['frame_lookup'].get(frame_number)

        if ref_landmarks is None or user_landmarks is None:
            return None

        score, detailed_scores = compare_poses(ref_landmarks, user_landmarks)

        return {
            'overall_score': round(score, 2),
            'angle_score': round(detailed_scores.get('angles', 0), 2),
            'orientation_score': round(detailed_scores.get('orientation', 0), 2),
            'extension_score': round(detailed_scores.get('extension', 0), 2)
        }


# Initialize service
pose_service = PoseAnalysisService(LANDMARKS_FILE)


@app.route('/')
def index():
    """Serve main application page"""
    return render_template('index.html',
                         video_duration=pose_service.video_info.get('total_frames', 0) / pose_service.video_info.get('fps', 30))


@app.route('/api/video-info')
def video_info():
    """Get video metadata"""
    return jsonify({
        'fps': pose_service.video_info.get('fps', 30),
        'total_frames': pose_service.video_info.get('total_frames', 0),
        'width': pose_service.video_info.get('width', 0),
        'height': pose_service.video_info.get('height', 0),
        'duration': pose_service.video_info.get('total_frames', 0) / pose_service.video_info.get('fps', 30)
    })


@app.route('/api/analyze-pose', methods=['POST'])
def analyze_pose():
    """Analyze a single pose from webcam frame"""
    data = request.json

    frame_data = data.get('frame')
    timestamp = data.get('timestamp', 0)

    # Calculate frame number from timestamp
    fps = pose_service.video_info.get('fps', 30)
    frame_number = int(timestamp * fps)

    # Extract landmarks from frame
    user_landmarks = pose_service.extract_landmarks(frame_data)

    # Score the pose
    score_data = pose_service.score_pose(user_landmarks, frame_number)

    if score_data is None:
        return jsonify({'error': 'Could not analyze pose'}), 400

    return jsonify({
        'frame_number': frame_number,
        'timestamp': timestamp,
        'scores': score_data,
        'has_reference': frame_number in pose_service.reference_landmarks['frame_lookup']
    })


@app.route('/api/batch-analyze', methods=['POST'])
def batch_analyze():
    """Analyze multiple frames in batch (for post-processing)"""
    data = request.json
    frames = data.get('frames', [])

    results = []
    fps = pose_service.video_info.get('fps', 30)

    for frame_data in frames:
        timestamp = frame_data.get('timestamp', 0)
        frame_number = int(timestamp * fps)

        user_landmarks = pose_service.extract_landmarks(frame_data.get('frame'))
        score_data = pose_service.score_pose(user_landmarks, frame_number)

        if score_data:
            results.append({
                'frame_number': frame_number,
                'timestamp': timestamp,
                'scores': score_data
            })

    return jsonify({
        'results': results,
        'total_analyzed': len(results)
    })


@app.route('/video/<path:filename>')
def serve_video(filename):
    """Serve video file"""
    import os
    video_path = os.path.abspath('.')
    return send_from_directory(video_path, filename, mimetype='video/mp4')


if __name__ == '__main__':
    # Check required files exist
    if not os.path.exists(VIDEO_FILE):
        print(f"Error: {VIDEO_FILE} not found")
        exit(1)
    if not os.path.exists(LANDMARKS_FILE):
        print(f"Error: {LANDMARKS_FILE} not found")
        exit(1)

    print("🎭 Dance Dance Live Server")
    print("=" * 50)
    print(f"Video: {VIDEO_FILE}")
    print(f"Landmarks: {LANDMARKS_FILE}")
    print(f"Video Info: {pose_service.video_info}")
    print("\nServer starting on http://localhost:5000")
    print("=" * 50)

    app.run(debug=True, host='0.0.0.0', port=5000)
