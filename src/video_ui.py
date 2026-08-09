import cv2 as cv


# ============================================================
# DRAW MESSAGE
# ============================================================

def draw_message(
    frame,
    text,
    width=190,
    height=48,
    background_color=(25, 25, 25),
    text_position=(20, 36),
    font=cv.FONT_HERSHEY_SIMPLEX,
    font_scale=0.5,
    text_color=(0, 220, 255),
    thickness=1,
    line_type=cv.LINE_AA
):

    cv.rectangle(
        frame,
        (10, 10),
        (width, height),
        background_color,
        -1
    )

    cv.putText(
        frame,
        text,
        text_position,
        font,
        font_scale,
        text_color,
        thickness,
        line_type
    )


# ============================================================
# DRAW PREDICTION
# ============================================================

def draw_prediction(
    frame,
    letter,
    confidence
):

    # --------------------------------
    # PREDICTION PANEL
    # --------------------------------

    cv.rectangle(
        frame,
        (10, 10),
        (190, 85),
        (25, 25, 25),
        -1
    )

    # --------------------------------
    # GREEN ACCENT BAR
    # --------------------------------

    cv.rectangle(
        frame,
        (10, 10),
        (14, 85),
        (80, 220, 120),
        -1
    )

    # --------------------------------
    # PREDICTION LABEL
    # --------------------------------

    cv.putText(
        frame,
        "PREDICTION",
        (24, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.42,
        (180, 180, 180),
        1,
        cv.LINE_AA
    )

    # --------------------------------
    # PREDICTED LETTER
    # --------------------------------

    cv.putText(
        frame,
        letter,
        (24, 68),
        cv.FONT_HERSHEY_SIMPLEX,
        0.95,
        (255, 255, 255),
        2,
        cv.LINE_AA
    )

    # --------------------------------
    # CONFIDENCE
    # --------------------------------

    cv.putText(
        frame,
        f"{confidence:.1%}",
        (70, 68),
        cv.FONT_HERSHEY_SIMPLEX,
        0.55,
        (80, 220, 120),
        1,
        cv.LINE_AA
    )