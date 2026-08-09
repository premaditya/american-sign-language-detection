import cv2 as cv
import mediapipe as mp
import numpy as np
import time

from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = str(
    Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"
)


BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode


# ============================================================
# IMAGE LANDMARKER
# ============================================================

image_options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=VisionRunningMode.IMAGE,

    num_hands=1
)


image_landmarker = HandLandmarker.create_from_options(
    image_options
)


# ============================================================
# VIDEO LANDMARKER
# ============================================================

video_options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=VisionRunningMode.VIDEO,

    num_hands=1
)


video_landmarker = HandLandmarker.create_from_options(
    video_options
)


START_TIME = time.perf_counter()


# ============================================================
# CONVERT MEDIAPIPE LANDMARKS TO FEATURES
# ============================================================

def _landmarks_to_features(result):

    if len(result.hand_landmarks) == 0:
        return None

    landmarks = result.hand_landmarks[0]

    wrist = landmarks[0]

    middle_base = landmarks[9]

    scale = np.sqrt(
        (wrist.x - middle_base.x) ** 2 +
        (wrist.y - middle_base.y) ** 2 +
        (wrist.z - middle_base.z) ** 2
    )

    if scale == 0:
        return None

    features = []

    for point in landmarks:

        new_landmark_1 = point.x - wrist.x
        new_landmark_2 = point.y - wrist.y
        new_landmark_3 = point.z - wrist.z

        new_landmark_1 = new_landmark_1 / scale
        new_landmark_2 = new_landmark_2 / scale
        new_landmark_3 = new_landmark_3 / scale

        features.extend([
            new_landmark_1,
            new_landmark_2,
            new_landmark_3
        ])

    return np.array(
        features,
        dtype=np.float32
    )


# ============================================================
# DRAW MEDIAPIPE LANDMARKS
# ============================================================

def _draw_landmarks(frame, result):

    if len(result.hand_landmarks) == 0:
        return

    height, width = frame.shape[:2]

    # MediaPipe hand connections
    connections = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),

        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),

        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),

        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),

        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),

        (5, 9),
        (9, 13),
        (13, 17)
    ]

    for hand_landmarks in result.hand_landmarks:

        points = []

        # --------------------------------
        # CONVERT NORMALIZED COORDINATES
        # TO PIXEL COORDINATES
        # --------------------------------

        for landmark in hand_landmarks:

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            points.append((x, y))

        # --------------------------------
        # DRAW CONNECTIONS
        # --------------------------------

        for start, end in connections:

            cv.line(
                frame,
                points[start],
                points[end],
                (0, 255, 0),
                2
            )

        # --------------------------------
        # DRAW LANDMARK POINTS
        # --------------------------------

        for i, (x, y) in enumerate(points):

            cv.circle(
                frame,
                (x, y),
                5,
                (0, 0, 255),
                -1
            )

            # Landmark number
            cv.putText(
                frame,
                str(i),
                (x + 7, y - 7),
                cv.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1
            )


# ============================================================
# IMAGE LANDMARK EXTRACTION
# ============================================================

def extract_landmarks_image(image):

    if image is None:
        return None

    rgb = cv.cvtColor(
        image,
        cv.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = image_landmarker.detect(
        mp_image
    )

    return _landmarks_to_features(result)


# ============================================================
# VIDEO LANDMARK EXTRACTION
# ============================================================

def extract_landmarks_video(frame, draw=False):

    if frame is None:
        return None

    rgb = cv.cvtColor(
        frame,
        cv.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp = int(
        (time.perf_counter() - START_TIME) * 1000
    )

    result = video_landmarker.detect_for_video(
        mp_image,
        timestamp
    )

    # --------------------------------
    # DRAW LANDMARKS IF REQUESTED
    # --------------------------------

    if draw:

        _draw_landmarks(
            frame,
            result
        )

    # --------------------------------
    # RETURN FEATURES FOR ANN
    # --------------------------------

    return _landmarks_to_features(result)