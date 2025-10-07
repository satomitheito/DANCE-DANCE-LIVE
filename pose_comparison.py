#!/usr/bin/env python3
"""
Pose Comparison Module
Provides functions to normalize and compare pose landmarks for dance matching
"""

import numpy as np
from typing import Dict, Optional, Tuple


# MediaPipe Pose Landmark Indices
class PoseLandmark:
    """MediaPipe pose landmark indices"""
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28


def calculate_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """Calculate Euclidean distance between two points (x, y)"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def calculate_angle(point1: np.ndarray, point2: np.ndarray, point3: np.ndarray) -> float:
    """
    Calculate angle at point2 formed by point1-point2-point3
    Returns angle in degrees (0-180)
    """
    # Create vectors
    vector1 = point1[:2] - point2[:2]  # Use only x, y
    vector2 = point3[:2] - point2[:2]

    # Calculate angle using dot product
    cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2) + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Avoid numerical errors
    angle = np.arccos(cos_angle)

    return np.degrees(angle)


def calculate_line_angle(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Calculate angle of line from point1 to point2 relative to horizontal
    Returns angle in degrees (-180 to 180)
    """
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    return np.degrees(np.arctan2(dy, dx))


def normalize_landmarks(landmarks: np.ndarray) -> Optional[np.ndarray]:
    """
    Normalize landmarks to be scale and position invariant
    Uses torso length (shoulder to hip) as normalization factor
    Centers the pose at the torso midpoint

    Args:
        landmarks: Array of shape (33, 3) with [x, y, z] coordinates

    Returns:
        Normalized landmarks or None if normalization fails
    """
    if landmarks is None or len(landmarks) < 33:
        return None

    try:
        # Get key points
        left_shoulder = landmarks[PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[PoseLandmark.LEFT_HIP]
        right_hip = landmarks[PoseLandmark.RIGHT_HIP]

        # Calculate torso center and length
        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2
        torso_length = calculate_distance(shoulder_center, hip_center)

        if torso_length < 1e-6:
            return None

        # Normalize: center at origin and scale by torso length
        normalized = landmarks.copy()

        # Center at torso midpoint (using only x, y for centering)
        torso_midpoint = (shoulder_center + hip_center) / 2
        normalized[:, 0] -= torso_midpoint[0]
        normalized[:, 1] -= torso_midpoint[1]

        # Scale by torso length
        normalized[:, 0] /= torso_length
        normalized[:, 1] /= torso_length
        normalized[:, 2] /= torso_length  # Also normalize z

        return normalized

    except Exception as e:
        print(f"Error normalizing landmarks: {e}")
        return None


def calculate_joint_angles(landmarks: np.ndarray) -> Dict[str, float]:
    """
    Calculate all major joint angles

    Args:
        landmarks: Array of shape (33, 3) with [x, y, z] coordinates

    Returns:
        Dictionary of joint angles in degrees
    """
    angles = {}

    try:
        # Left elbow angle
        angles['left_elbow'] = calculate_angle(
            landmarks[PoseLandmark.LEFT_SHOULDER],
            landmarks[PoseLandmark.LEFT_ELBOW],
            landmarks[PoseLandmark.LEFT_WRIST]
        )

        # Right elbow angle
        angles['right_elbow'] = calculate_angle(
            landmarks[PoseLandmark.RIGHT_SHOULDER],
            landmarks[PoseLandmark.RIGHT_ELBOW],
            landmarks[PoseLandmark.RIGHT_WRIST]
        )

        # Left shoulder angle
        angles['left_shoulder'] = calculate_angle(
            landmarks[PoseLandmark.LEFT_HIP],
            landmarks[PoseLandmark.LEFT_SHOULDER],
            landmarks[PoseLandmark.LEFT_ELBOW]
        )

        # Right shoulder angle
        angles['right_shoulder'] = calculate_angle(
            landmarks[PoseLandmark.RIGHT_HIP],
            landmarks[PoseLandmark.RIGHT_SHOULDER],
            landmarks[PoseLandmark.RIGHT_ELBOW]
        )

        # Left knee angle
        angles['left_knee'] = calculate_angle(
            landmarks[PoseLandmark.LEFT_HIP],
            landmarks[PoseLandmark.LEFT_KNEE],
            landmarks[PoseLandmark.LEFT_ANKLE]
        )

        # Right knee angle
        angles['right_knee'] = calculate_angle(
            landmarks[PoseLandmark.RIGHT_HIP],
            landmarks[PoseLandmark.RIGHT_KNEE],
            landmarks[PoseLandmark.RIGHT_ANKLE]
        )

        # Left hip angle
        angles['left_hip'] = calculate_angle(
            landmarks[PoseLandmark.LEFT_SHOULDER],
            landmarks[PoseLandmark.LEFT_HIP],
            landmarks[PoseLandmark.LEFT_KNEE]
        )

        # Right hip angle
        angles['right_hip'] = calculate_angle(
            landmarks[PoseLandmark.RIGHT_SHOULDER],
            landmarks[PoseLandmark.RIGHT_HIP],
            landmarks[PoseLandmark.RIGHT_KNEE]
        )

    except Exception as e:
        print(f"Error calculating joint angles: {e}")

    return angles


def calculate_body_orientation(landmarks: np.ndarray) -> Dict[str, float]:
    """
    Calculate body orientation metrics

    Args:
        landmarks: Array of shape (33, 3) with [x, y, z] coordinates

    Returns:
        Dictionary of orientation angles in degrees
    """
    orientation = {}

    try:
        # Shoulder line angle (tilt)
        orientation['shoulder_tilt'] = calculate_line_angle(
            landmarks[PoseLandmark.LEFT_SHOULDER],
            landmarks[PoseLandmark.RIGHT_SHOULDER]
        )

        # Hip line angle (tilt)
        orientation['hip_tilt'] = calculate_line_angle(
            landmarks[PoseLandmark.LEFT_HIP],
            landmarks[PoseLandmark.RIGHT_HIP]
        )

        # Torso angle (vertical tilt)
        shoulder_center = (landmarks[PoseLandmark.LEFT_SHOULDER] + landmarks[PoseLandmark.RIGHT_SHOULDER]) / 2
        hip_center = (landmarks[PoseLandmark.LEFT_HIP] + landmarks[PoseLandmark.RIGHT_HIP]) / 2
        orientation['torso_tilt'] = calculate_line_angle(hip_center, shoulder_center)

        # Torso twist (using z-coordinate difference between shoulders)
        shoulder_z_diff = abs(landmarks[PoseLandmark.LEFT_SHOULDER][2] - landmarks[PoseLandmark.RIGHT_SHOULDER][2])
        orientation['torso_twist'] = shoulder_z_diff

    except Exception as e:
        print(f"Error calculating body orientation: {e}")

    return orientation


def calculate_limb_extension(landmarks: np.ndarray) -> Dict[str, float]:
    """
    Calculate limb extension ratios

    Args:
        landmarks: Array of shape (33, 3) with [x, y, z] coordinates

    Returns:
        Dictionary of extension ratios
    """
    ratios = {}

    try:
        # Left arm extension (wrist distance from shoulder relative to full arm length)
        left_upper_arm = calculate_distance(
            landmarks[PoseLandmark.LEFT_SHOULDER],
            landmarks[PoseLandmark.LEFT_ELBOW]
        )
        left_forearm = calculate_distance(
            landmarks[PoseLandmark.LEFT_ELBOW],
            landmarks[PoseLandmark.LEFT_WRIST]
        )
        left_arm_length = left_upper_arm + left_forearm
        left_extension_dist = calculate_distance(
            landmarks[PoseLandmark.LEFT_SHOULDER],
            landmarks[PoseLandmark.LEFT_WRIST]
        )
        ratios['left_arm_extension'] = left_extension_dist / (left_arm_length + 1e-6)

        # Right arm extension
        right_upper_arm = calculate_distance(
            landmarks[PoseLandmark.RIGHT_SHOULDER],
            landmarks[PoseLandmark.RIGHT_ELBOW]
        )
        right_forearm = calculate_distance(
            landmarks[PoseLandmark.RIGHT_ELBOW],
            landmarks[PoseLandmark.RIGHT_WRIST]
        )
        right_arm_length = right_upper_arm + right_forearm
        right_extension_dist = calculate_distance(
            landmarks[PoseLandmark.RIGHT_SHOULDER],
            landmarks[PoseLandmark.RIGHT_WRIST]
        )
        ratios['right_arm_extension'] = right_extension_dist / (right_arm_length + 1e-6)

        # Left leg extension
        left_thigh = calculate_distance(
            landmarks[PoseLandmark.LEFT_HIP],
            landmarks[PoseLandmark.LEFT_KNEE]
        )
        left_shin = calculate_distance(
            landmarks[PoseLandmark.LEFT_KNEE],
            landmarks[PoseLandmark.LEFT_ANKLE]
        )
        left_leg_length = left_thigh + left_shin
        left_leg_extension_dist = calculate_distance(
            landmarks[PoseLandmark.LEFT_HIP],
            landmarks[PoseLandmark.LEFT_ANKLE]
        )
        ratios['left_leg_extension'] = left_leg_extension_dist / (left_leg_length + 1e-6)

        # Right leg extension
        right_thigh = calculate_distance(
            landmarks[PoseLandmark.RIGHT_HIP],
            landmarks[PoseLandmark.RIGHT_KNEE]
        )
        right_shin = calculate_distance(
            landmarks[PoseLandmark.RIGHT_KNEE],
            landmarks[PoseLandmark.RIGHT_ANKLE]
        )
        right_leg_length = right_thigh + right_shin
        right_leg_extension_dist = calculate_distance(
            landmarks[PoseLandmark.RIGHT_HIP],
            landmarks[PoseLandmark.RIGHT_ANKLE]
        )
        ratios['right_leg_extension'] = right_leg_extension_dist / (right_leg_length + 1e-6)

    except Exception as e:
        print(f"Error calculating limb extensions: {e}")

    return ratios


def compare_angles(angles1: Dict[str, float], angles2: Dict[str, float],
                   max_angle_diff: float = 180.0) -> float:
    """
    Compare two sets of joint angles

    Args:
        angles1: First set of angles
        angles2: Second set of angles
        max_angle_diff: Maximum possible angle difference (for normalization)

    Returns:
        Similarity score between 0 and 1
    """
    if not angles1 or not angles2:
        return 0.0

    similarities = []

    for key in angles1:
        if key in angles2:
            angle_diff = abs(angles1[key] - angles2[key])
            similarity = 1.0 - (angle_diff / max_angle_diff)
            similarities.append(max(0.0, similarity))

    return np.mean(similarities) if similarities else 0.0


def compare_orientations(orient1: Dict[str, float], orient2: Dict[str, float],
                        max_tilt_diff: float = 180.0, max_twist_diff: float = 1.0) -> float:
    """
    Compare two sets of body orientations

    Args:
        orient1: First set of orientations
        orient2: Second set of orientations
        max_tilt_diff: Maximum tilt angle difference
        max_twist_diff: Maximum twist difference

    Returns:
        Similarity score between 0 and 1
    """
    if not orient1 or not orient2:
        return 0.0

    similarities = []

    # Compare tilt angles
    for key in ['shoulder_tilt', 'hip_tilt', 'torso_tilt']:
        if key in orient1 and key in orient2:
            angle_diff = abs(orient1[key] - orient2[key])
            # Handle wrap-around for angles
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            similarity = 1.0 - (angle_diff / max_tilt_diff)
            similarities.append(max(0.0, similarity))

    # Compare torso twist
    if 'torso_twist' in orient1 and 'torso_twist' in orient2:
        twist_diff = abs(orient1['torso_twist'] - orient2['torso_twist'])
        similarity = 1.0 - (twist_diff / max_twist_diff)
        similarities.append(max(0.0, similarity))

    return np.mean(similarities) if similarities else 0.0


def compare_extensions(ext1: Dict[str, float], ext2: Dict[str, float],
                       max_extension_diff: float = 1.0) -> float:
    """
    Compare limb extension ratios

    Args:
        ext1: First set of extensions
        ext2: Second set of extensions
        max_extension_diff: Maximum extension ratio difference

    Returns:
        Similarity score between 0 and 1
    """
    if not ext1 or not ext2:
        return 0.0

    similarities = []

    for key in ext1:
        if key in ext2:
            ext_diff = abs(ext1[key] - ext2[key])
            similarity = 1.0 - (ext_diff / max_extension_diff)
            similarities.append(max(0.0, similarity))

    return np.mean(similarities) if similarities else 0.0


def compare_poses(ref_landmarks: np.ndarray, user_landmarks: np.ndarray,
                 weights: Optional[Dict[str, float]] = None) -> Tuple[float, Dict[str, float]]:
    """
    Compare two poses and return overall similarity score

    Args:
        ref_landmarks: Reference pose landmarks (33, 3)
        user_landmarks: User pose landmarks (33, 3)
        weights: Optional custom weights for different metrics
                Default: {'angles': 0.6, 'orientation': 0.3, 'extension': 0.1}

    Returns:
        Tuple of (overall_score, detailed_scores)
        overall_score: Float between 0-100
        detailed_scores: Dictionary with individual metric scores
    """
    # Default weights
    if weights is None:
        weights = {
            'angles': 0.6,
            'orientation': 0.3,
            'extension': 0.1
        }

    # Normalize both poses
    ref_norm = normalize_landmarks(ref_landmarks)
    user_norm = normalize_landmarks(user_landmarks)

    if ref_norm is None or user_norm is None:
        return 0.0, {}

    # Calculate metrics for both poses
    ref_angles = calculate_joint_angles(ref_norm)
    user_angles = calculate_joint_angles(user_norm)

    ref_orient = calculate_body_orientation(ref_norm)
    user_orient = calculate_body_orientation(user_norm)

    ref_ext = calculate_limb_extension(ref_norm)
    user_ext = calculate_limb_extension(user_norm)

    # Compare metrics
    angle_similarity = compare_angles(ref_angles, user_angles)
    orientation_similarity = compare_orientations(ref_orient, user_orient)
    extension_similarity = compare_extensions(ref_ext, user_ext)

    # Calculate weighted overall score
    overall_score = (
        angle_similarity * weights['angles'] +
        orientation_similarity * weights['orientation'] +
        extension_similarity * weights['extension']
    ) * 100.0

    # Detailed scores
    detailed_scores = {
        'overall': overall_score,
        'angles': angle_similarity * 100.0,
        'orientation': orientation_similarity * 100.0,
        'extension': extension_similarity * 100.0,
        'angle_details': {k: abs(ref_angles.get(k, 0) - user_angles.get(k, 0)) for k in ref_angles if k in user_angles},
        'orientation_details': {k: abs(ref_orient.get(k, 0) - user_orient.get(k, 0)) for k in ref_orient if k in user_orient}
    }

    return overall_score, detailed_scores
