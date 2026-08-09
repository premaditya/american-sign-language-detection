import numpy as np
import cv2 as cv

from src.mediapipe_landmarks import extract_landmarks_image
from src.predictor import prediction


def predict_image(image):

    image = np.array(image)

    image = cv.cvtColor(
        image,
        cv.COLOR_RGB2BGR
    )

    landmarks = extract_landmarks_image(
        image
    )

    if landmarks is None:
        return None, "no_hand"

    top_5_pred = prediction(
        landmarks
    )

    if top_5_pred is None:
        return None, "prediction_failed"

    return top_5_pred, None