import streamlit as st

from PIL import Image

from src.image_predictor import predict_image
#from src.video_processor import VideoProcessor

#from streamlit_webrtc import webrtc_streamer


st.set_page_config(
    page_title= "American Sign Language Detection",
    page_icon= "🤘",
    layout= "wide"
)


with open("app.css") as f:
    css = f.read()

st.markdown(
    f"""
    <style>
    {css}
    </style>
    """,
    unsafe_allow_html=True
)

def confidence_color(confidence):

    if confidence < 50:
        return "#ff4b4b"

    elif confidence < 80:
        return "#ffc107"

    else:
        return "#50dc78"

def display_prediction(top_5_pred):

    pred_letter, pred_confidence = top_5_pred[0]
    
    pred_confidence *= 100

    pred_color = confidence_color(
    pred_confidence
)

    # --------------------------------
    # MAIN PREDICTION
    # --------------------------------

    st.markdown(f"""
    <div class="pred-main-card">
        <div class="pred-main-header">
            Prediction: {pred_letter}
        </div>
        <div class="progress-track">
            <div class="progress-fill"
                style="width:{pred_confidence:.2f}%;background-color:{pred_color};">
            </div>
        </div>
        <div class="progress-label">
            {pred_confidence:.2f}%
        </div>
    </div>
    """,unsafe_allow_html=True)


    # --------------------------------
    # TOP-5 SUB PREDICTIONS
    # --------------------------------

    for letter, confidence in top_5_pred[1:]:

        confidence *= 100

        pred_color = confidence_color(
            confidence
        )

        if confidence >= 2:

            st.markdown(f"""
            <div class="pred-sub-row">
                <div class="pred-sub-letter">
                    {letter}
                </div>
                <div class="progress-track-sub">
                    <div class="progress-fill-sub"
                        style="width:{confidence:.2f}%; background-color:{pred_color};">
                    </div>
                </div>
                <div class="progress-label-sub">
                    {confidence:.2f}%
                </div>
            </div>
            """,unsafe_allow_html=True)


with st.sidebar:

    # --------------------------------
    # SIDEBAR BRANDING / HEADER
    # --------------------------------

    st.markdown("""
    <div class="sidebar-header">
        <h2>🤘 ASL Detector</h2>
        <p class="sidebar-subtitle">Sign language, decoded live</p>
    </div>
    """, unsafe_allow_html=True)


    st.markdown(
        '<hr class="sidebar-divider">',
        unsafe_allow_html=True
    )


    # --------------------------------
    # NAVIGATION
    # --------------------------------

    page = st.radio(
        "Select Page",
        ["Home", "Image", "Webcam"]
    )


    st.markdown(
        '<hr class="sidebar-divider">',
        unsafe_allow_html=True
    )


    # --------------------------------
    # EXTRA SIDEBAR CONTENT
    # --------------------------------

    st.markdown("""
    <div class="sidebar-footer">
        <p class="sidebar-footer-text">
            Model Accuracy:
            <span class="accent">98.59%</span>
        </p>
        <p class="sidebar-footer-text">
            Built with MediaPipe + ANN
        </p>
    </div>
    """,unsafe_allow_html=True)


    # --------------------------------
    # AUTHOR INFO
    # --------------------------------

    st.markdown("""
    <div class="sidebar-author">
        <div class="sidebar-author-avatar">
            PA
        </div>
        <p class="sidebar-author-name">
            Dhulipala Prem Aditya
        </p>
        <div class="sidebar-author-links">
            <a href="https://www.linkedin.com/in/prem-aditya-dhulipala-627a43268"
                target="_blank"
                class="sidebar-link">
                LinkedIn
            </a>
            <a href="https://github.com/premaditya"
                target="_blank"
                class="sidebar-link">GitHub
            </a>
        </div>
    </div>
    """,unsafe_allow_html=True)



if page == "Home":

    c1, c2, c3 = st.columns([1,4,1])


    with c2:

        st.markdown("""
        <div class = "header">
        <h1>🤘 American Sign Language Detection</h1>

        </div>
        """,unsafe_allow_html= True)


    st.markdown("---")


    st.markdown("""
    <div class="desc">

    This is an ANN Deep Learning model that helps detect American Sign Language hand signs. The model is trained on 1,000 images per class from the ASL alphabet dataset.<br>

    The training dataset is well balanced and no class dominates another. MediaPipe Hand Landmarks are used to extract features from the images, and the model is trained on those landmarks.<br>

    Due to the complexity of fingers overlapping in some signs like M, N, Q, C, and X, landmark detection is less reliable for those classes, resulting in some class imbalance in the extracted feature data.<br>

    The baseline ANN model achieved a validation accuracy of around 96.67%, which was improved to approximately 97.13% validation accuracy using Hyperparameter Tuning with Optuna. After increasing the dataset to 1,000 images per class, the final model achieved a Test Accuracy of 98.59%.

    </div>
    """, unsafe_allow_html=True)


    st.markdown("""
    <div class = "desc_header">
    <h2>How to Use</h2>
    </div>
    """,unsafe_allow_html= True)


    st.markdown("""
    <div class="usage">

    First, in the sidebar, select whether you want to upload an image or use the webcam for live predictions.<br>

    <strong>Image Option</strong>

    <ul>

    <li>After selecting the Image option, choose either Upload Image or Take Picture.</li>

    <li>For Upload Image, select the image you want to use and click Open.</li>

    <li>For Take Picture, allow camera access and capture a clear picture of the hand sign.</li>

    <li>Then click the Predict button.</li>

    <li>You can now see your prediction along with the confidence score and top predictions.</li>

    </ul>


    <strong>Webcam Option</strong>

    <ul>

    <li>After selecting the Webcam option, click Start to access your webcam.</li>

    <li>Your browser will ask for permission to access the webcam — allow it.</li>

    <li>Show one ASL hand sign clearly to the webcam.</li>

    <li>Hold the sign steady while the system stabilizes the hand landmarks.</li>

    <li>Once the landmarks are stable, the model will make a prediction.</li>

    <li>The prediction will remain locked until you remove your hand from the webcam.</li>

    </ul>

    </div>
    """, unsafe_allow_html=True)


    st.markdown("""
    <div class="inst-header">
    <h3>Instructions</h3>
    </div>
    """, unsafe_allow_html=True)


    st.markdown("""
    <div class="inst">

    <ul>

    <li>Whether using the Image or Webcam option, MediaPipe detects only one hand at a time.</li>

    <li>Make sure the sign is visible clearly and as accurately as possible.</li>

    <li>Make sure your entire hand is visible inside the camera frame.</li>

    <li>When using the webcam, hold the sign steady while the system stabilizes the hand landmarks.</li>

    <li>Good lighting and a clear background can help improve landmark detection.</li>

    <li>After a prediction is made, the prediction remains locked until the hand is removed from the webcam.</li>

    <li>Some predictions may be wrong — try again with a different angle, lighting, or background if this happens.</li>

    </ul>

    </div>
    """, unsafe_allow_html=True)



elif page == "Image":

    st.title("Upload Image/Take Picture")

    st.markdown("---")


    mode = st.radio(
        "Select Mode",
        ["Upload Image","Take Picture"]
    )


    if mode == "Upload Image":

        uploaded_file = st.file_uploader(
            "Upload an Image",
            type=["jpg", "jpeg", "png"]
        )


        if uploaded_file is not None:

            if st.button("Predict"):

                image = Image.open(
                    uploaded_file
                )

                st.image(
                    image,
                    caption="Uploaded Image",
                    use_container_width=True
                )

                top_5_pred, error = predict_image(image)

                if error == "no_hand":

                    st.warning(
                        "No hand detected. Please place your hand clearly in the camera."
                    )

                elif error == "prediction_failed":

                    st.error(
                        "Prediction failed."
                    )

                else:

                    display_prediction(top_5_pred)


    else:

        camera_image = st.camera_input(
            "Take a Picture"
        )

        if camera_image is not None:

            if st.button("Predict"):

                image = Image.open(
                    camera_image
                )

                top_5_pred, error = predict_image(image)

                if error == "no_hand":

                    st.warning(
                        "No hand detected. Please place your hand clearly in the camera."
                    )

                elif error == "prediction_failed":

                    st.error(
                        "Prediction failed."
                    )

                else:

                    display_prediction(top_5_pred)



else:

    st.title("Click on START for Video")

    st.wraning("Testing if the webcam is causing the app to crash"

    """webrtc_streamer(

        key="asl-webcam",

        media_stream_constraints={

            "video": {

                "width": {
                    "ideal": 1280
                },

                "height": {
                    "ideal": 720
                },

                "frameRate": {
                    "ideal": 30
                }

            },

            "audio": False

        },

        rtc_configuration={
        "iceServers": [
            {
                "urls": ["stun:stun.l.google.com:19302"]
            }
        ]
        },

        video_processor_factory=VideoProcessor

    )"""
