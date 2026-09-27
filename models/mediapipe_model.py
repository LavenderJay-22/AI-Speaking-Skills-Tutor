import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

from config import MEDIAPIPE_MODEL_PATH, MEDIAPIPE_MODEL_URL

def download_model():
    """
    Download the FaceLandmarker model file if not already present.
    """
    model_path = MEDIAPIPE_MODEL_PATH
    dir_name = os.path.dirname(model_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    if not os.path.exists(model_path):
        print("Downloading FaceLandmarker model...")
        urllib.request.urlretrieve(MEDIAPIPE_MODEL_URL, model_path)
        print("Model downloaded.")
    return model_path

def analyse_gaze(video_path: str) -> dict:
    """
    Analyse gaze direction from a video file using MediaPipe FaceLandmarker.
    Processes every 10th frame for efficiency on consumer hardware.
    Returns a summary of gaze direction and eye contact estimate.
    """
    try:
        model_path = download_model()
    except Exception as e:
        return {
            "error": f"Could not download model: {str(e)}",
            "gaze_summary": "unavailable",
            "eye_contact_ratio": 0.0,
            "frames_analysed": 0
        }
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        return {
            "error": "Could not open video file",
            "gaze_summary": "unavailable",
            "eye_contact_ratio": 0.0,
            "frames_analysed": 0
        }

    fps = video_capture.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    gaze_directions = []
    head_yaw_ratios = []
    frames_analysed = 0
    frame_index = 0

    print(f"Analysing gaze from video: {video_path}")
    print(f"Total frames: {total_frames}, FPS: {fps:.1f}")

    base_options = python.BaseOptions(model_asset_path=model_path)
    landmarker_options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )

    try:
        with vision.FaceLandmarker.create_from_options(landmarker_options) as face_landmarker:
            while video_capture.isOpened():
                frame_read_ok, frame = video_capture.read()
                if not frame_read_ok:
                    break

                # Sample every 10th frame
                if frame_index % 10 == 0:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb_frame
                    )
                    detection_result = face_landmarker.detect(mp_image)

                    if detection_result.face_landmarks:
                        face_landmarks = detection_result.face_landmarks[0]
                        gaze_direction = estimate_gaze(face_landmarks, frame.shape)
                        gaze_directions.append(gaze_direction)
                        head_yaw_ratios.append(estimate_head_pose(face_landmarks))
                        frames_analysed += 1

                frame_index += 1

    except Exception as e:
        video_capture.release()
        return {
            "error": f"Gaze analysis failed: {str(e)}",
            "gaze_summary": "unavailable",
            "eye_contact_ratio": 0.0,
            "frames_analysed": 0
        }

    video_capture.release()

    if not gaze_directions:
        return {
            "error": "No face detected in video",
            "gaze_summary": "no face detected",
            "eye_contact_ratio": 0.0,
            "frames_analysed": 0
        }

    # Tally gaze directions
    centre_count = gaze_directions.count("centre")
    left_count = gaze_directions.count("left")
    right_count = gaze_directions.count("right")
    up_count = gaze_directions.count("up")
    down_count = gaze_directions.count("down")
    total_gaze_samples = len(gaze_directions)

    eye_contact_ratio = round(centre_count / total_gaze_samples, 2)

    if eye_contact_ratio >= 0.6:
        gaze_summary = "strong eye contact - looked at camera most of the time"
    elif eye_contact_ratio >= 0.4:
        gaze_summary = "moderate eye contact - looked away occasionally"
    elif eye_contact_ratio >= 0.2:
        gaze_summary = "limited eye contact - looked away frequently"
    else:
        gaze_summary = "minimal eye contact - rarely looked at camera"

    print(f"Gaze analysis complete. Frames analysed: {frames_analysed}")
    print(f"Eye contact ratio: {eye_contact_ratio}")
    print(f"Gaze summary: {gaze_summary}")

    head_orientation = classify_head_orientation(head_yaw_ratios)

    return {
        "gaze_summary": gaze_summary,
        "eye_contact_ratio": eye_contact_ratio,
        "frames_analysed": frames_analysed,
        "head_orientation": head_orientation,
        "gaze_breakdown": {
            "centre": centre_count,
            "left": left_count,
            "right": right_count,
            "up": up_count,
            "down": down_count
        }
    }

def estimate_gaze(face_landmarks, frame_shape) -> str:
    """
    Estimate gaze direction from facial landmarks.
    Uses both left and right iris landmarks for gaze estimation.
    Returns one of: centre, left, right, up, down
    """
    frame_height, frame_width = frame_shape[:2]

    try:
        # Iris landmark groups in the MediaPipe Tasks API
        LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
        RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]

        # Left eye corner landmarks
        left_eye_left_x = face_landmarks[33].x * frame_width
        left_eye_right_x = face_landmarks[133].x * frame_width
        left_eye_top_y = face_landmarks[159].y * frame_height
        left_eye_bottom_y = face_landmarks[145].y * frame_height

        # Right eye corner landmarks
        right_eye_left_x = face_landmarks[362].x * frame_width
        right_eye_right_x = face_landmarks[263].x * frame_width
        right_eye_top_y = face_landmarks[386].y * frame_height
        right_eye_bottom_y = face_landmarks[374].y * frame_height

        # Calculate left iris centre
        left_iris_x = np.mean([face_landmarks[i].x for i in LEFT_IRIS_INDICES]) * frame_width
        left_iris_y = np.mean([face_landmarks[i].y for i in LEFT_IRIS_INDICES]) * frame_height

        # Calculate right iris centre
        right_iris_x = np.mean([face_landmarks[i].x for i in RIGHT_IRIS_INDICES]) * frame_width
        right_iris_y = np.mean([face_landmarks[i].y for i in RIGHT_IRIS_INDICES]) * frame_height

        # Calculate gaze position within each eye
        left_eye_width = left_eye_right_x - left_eye_left_x
        left_eye_height = left_eye_bottom_y - left_eye_top_y

        right_eye_width = right_eye_right_x - right_eye_left_x
        right_eye_height = right_eye_bottom_y - right_eye_top_y

        left_horizontal_ratio = (
            (left_iris_x - left_eye_left_x) / left_eye_width
            if left_eye_width > 0 else 0.5
        )
        left_vertical_ratio = (
            (left_iris_y - left_eye_top_y) / left_eye_height
            if left_eye_height > 0 else 0.5
        )

        right_horizontal_ratio = (
            (right_iris_x - right_eye_left_x) / right_eye_width
            if right_eye_width > 0 else 0.5
        )
        right_vertical_ratio = (
            (right_iris_y - right_eye_top_y) / right_eye_height
            if right_eye_height > 0 else 0.5
        )

        # Average both eyes to obtain the overall gaze position
        horizontal_ratio = (left_horizontal_ratio + right_horizontal_ratio) / 2
        vertical_ratio = (left_vertical_ratio + right_vertical_ratio) / 2

        if horizontal_ratio < 0.35:
            return "right"
        elif horizontal_ratio > 0.65:
            return "left"
        elif vertical_ratio < 0.35:
            return "up"
        elif vertical_ratio > 0.65:
            return "down"
        else:
            return "centre"

    except Exception:
        return "centre"

def estimate_head_pose(face_landmarks) -> float:
    """
    Rough horizontal head orientation (yaw) estimate, using the
    nose tip position relative to the left/right face boundary
    landmarks. Returns a signed ratio: near 0 = facing forward,
    larger magnitude = turned toward one side.
    """
    nose_landmark = face_landmarks[1]
    left_face_landmark = face_landmarks[234]
    right_face_landmark = face_landmarks[454]

    face_width = right_face_landmark.x - left_face_landmark.x
    nose_offset = nose_landmark.x - ((left_face_landmark.x + right_face_landmark.x) / 2)

    return nose_offset / face_width if face_width > 0 else 0.0

def classify_head_orientation(yaw_ratios: list) -> dict:
    """
    Given per-frame yaw ratios, determine what proportion of the
    response was spent facing the camera versus turned away to
    either side.
    """
    TURNED_AWAY_THRESHOLD = 0.15

    if len(yaw_ratios) < 5:
        return {
            "forward_pct": None,
            "orientation": "insufficient data",
            "summary": "Not enough frames to assess head orientation."
        }

    # Count turned-away frames
    turned_away_count = sum(
        1 for yaw_ratio in yaw_ratios if abs(yaw_ratio) >= TURNED_AWAY_THRESHOLD
    )
    forward_pct = round(
        (1 - (turned_away_count / len(yaw_ratios))) * 100
    )

    if forward_pct >= 80:
        orientation = "forward-facing"
        summary = "The head was facing the camera for most of the response."
    elif forward_pct >= 60:
        orientation = "mostly forward"
        summary = "The head was turned away from the camera occasionally."
    else:
        orientation = "frequently turned away"
        summary = "The head was turned away from the camera for a large part of the response."
    return {
        "forward_pct": forward_pct,
        "orientation": orientation,
        "summary": summary
    }