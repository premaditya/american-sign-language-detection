import cv2 as cv
import av

from collections import deque
from streamlit_webrtc import VideoProcessorBase

from src.mediapipe_landmarks import extract_landmarks_video
from src.predictor import prediction
from src.video_ui import draw_message, draw_prediction


# ============================================================
# CHECK LANDMARK STABILITY
# ============================================================

def landmarks_are_similar(
    buffer,
    threshold=0.2
):

    if len(buffer) < 7:
        return False

    distances = []

    for i in range(1, len(buffer)):

        distance = (
            (buffer[i] - buffer[i - 1]) ** 2
        ).sum() ** 0.5

        distances.append(distance)

    max_distance = max(distances)

    if max_distance > threshold:
        return False

    return True


# ============================================================
# VIDEO PROCESSOR
# ============================================================

class VideoProcessor(VideoProcessorBase):

    def __init__(self):

        self.landmark_buffer = deque(maxlen=7)

        self.stable = False

        self.prediction = False
        self.pred_letter = ""
        self.pred_confidence = 0.0

    # ========================================================
    # RECEIVE VIDEO FRAME
    # ========================================================

    def recv(self, frame):

        try:

            img = frame.to_ndarray(
                format="bgr24"
            )

            processed_frame = self.predict_frame(
                img
            )

            return processed_frame

        except Exception as e:

            print("ERROR:", e)

            return frame

    # ========================================================
    # PROCESS FRAME
    # ========================================================

    def predict_frame(self, frame):

        # --------------------------------
        # EXTRACT LANDMARKS + DRAW THEM
        # --------------------------------

        landmarks = extract_landmarks_video(
            frame,
            draw=True
        )

        # --------------------------------
        # NO HAND DETECTED
        # --------------------------------

        if landmarks is None:

            self.landmark_buffer.clear()

            self.stable = False
            self.prediction = False
            self.pred_letter = ""
            self.pred_confidence = 0.0

            draw_message(
                frame,
                text="No hand detected",
                width=230,
                text_color=(80, 180, 255)
            )

            return av.VideoFrame.from_ndarray(
                frame,
                format="bgr24"
            )

        # --------------------------------
        # HAND DETECTED
        # --------------------------------

        if not self.prediction:

            self.landmark_buffer.append(
                landmarks
            )

            # --------------------------------
            # NOT ENOUGH FRAMES
            # --------------------------------

            if len(self.landmark_buffer) < 7:

                draw_message(
                    frame,
                    text="Hold still...",
                    width=190,
                    text_color=(0, 220, 255)
                )

            # --------------------------------
            # 7 FRAMES AVAILABLE
            # --------------------------------

            else:

                if landmarks_are_similar(
                    self.landmark_buffer
                ):

                    self.stable = True

                    draw_message(
                        frame,
                        text="Predicting...",
                        width=190,
                        text_color=(80, 220, 120)
                    )

                    top_5_pred = prediction(
                        landmarks
                    )

                    if top_5_pred is not None:

                        pred = top_5_pred[0]

                        self.pred_letter = pred[0]
                        self.pred_confidence = pred[1]

                        self.prediction = True

                    else:

                        draw_message(
                            frame,
                            text="Prediction failed",
                            width=210,
                            text_color=(80, 80, 255)
                        )

                # --------------------------------
                # LANDMARKS NOT STABLE
                # --------------------------------

                else:

                    self.stable = False

                    draw_message(
                        frame,
                        text="Hold still...",
                        width=190,
                        text_color=(0, 220, 255)
                    )

        # --------------------------------
        # DISPLAY LOCKED PREDICTION
        # --------------------------------

        if self.prediction:

            draw_prediction(
                frame,
                letter=self.pred_letter,
                confidence=self.pred_confidence
            )

        return av.VideoFrame.from_ndarray(
            frame,
            format="bgr24"
        )