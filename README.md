# 🤘 American Sign Language Detection
 
A real-time American Sign Language (ASL) alphabet detection application built using **MediaPipe Hand Landmarks** and an **Artificial Neural Network (ANN)**.
 
The application can detect ASL hand signs from uploaded images, captured pictures, and a live webcam.
 
## 📌 Overview
 
This project uses MediaPipe to extract 3D hand landmarks from an image or video frame. These landmarks are normalized and used as features for an Artificial Neural Network trained to classify ASL alphabet signs.
 
The application provides three modes:
 
- 🏠 **Home** – Project information and instructions
- 🖼️ **Image** – Upload an image or take a picture
- 📹 **Webcam** – Real-time ASL detection using the webcam
The final ANN model achieved a **98.59% test accuracy**.
 
## ✨ Features
 
- Real-time ASL alphabet detection
- Image upload prediction
- Camera image prediction
- Live webcam prediction
- MediaPipe hand landmark detection
- Normalized landmark-based features
- ANN deep learning classification
- Top-5 prediction results
- Confidence scores
- Confidence-based prediction bars
- Webcam landmark visualization
- Streamlit web interface
## 🧠 Machine Learning Pipeline
 
```text
ASL Images
    ↓
MediaPipe Hand Landmarks
    ↓
Landmark Normalization
    ↓
Feature Extraction
    ↓
ANN Model
    ↓
Prediction Probabilities
    ↓
Top-5 Predictions
```
 
### Feature Extraction
 
MediaPipe detects **21 hand landmarks**.
 
Each landmark contains:
 
- X coordinate
- Y coordinate
- Z coordinate
The landmarks are normalized relative to the wrist and scaled using the distance between the wrist and the middle finger base.
 
The resulting feature vector is used as input to the ANN model.
 
## 📊 Model Performance
 
| Model | Accuracy |
|---|---:|
| Baseline ANN | ~96.67% Validation Accuracy |
| Optuna Tuned ANN | ~97.13% Validation Accuracy |
| Final Model | **98.59% Test Accuracy** |
 
The final model was trained using **1,000 images per ASL alphabet class**.
 
## 🛠️ Technologies Used
 
- **Python**
- **TensorFlow / Keras**
- **MediaPipe**
- **OpenCV**
- **Scikit-learn**
- **NumPy**
- **Pandas**
- **Streamlit**
- **Streamlit-WebRTC**
- **Optuna**
- **Matplotlib**
- **Seaborn**
- **Pillow**
## 📁 Project Structure
 
```text
ASL/
│
├── models/
│   ├── final_ann_new.keras
│   ├── hand_landmarker.task
│   ├── label_encoder_new.pkl
│   └── scaler_new.pkl
│
├── src/
│   ├── image_predictor.py
│   ├── mediapipe_landmarks.py
│   ├── predictor.py
│   ├── video_processor.py
│   └── video_ui.py
│
├── app.py
├── app.css
├── asl_ann.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```
 
The datasets and virtual environment are excluded from the repository using `.gitignore`.
 
## 🚀 Installation
 
### 1. Clone the repository
 
```bash
git clone https://github.com/premaditya/ASL.git
```
 
### 2. Navigate to the project directory
 
```bash
cd ASL
```
 
### 3. Create a virtual environment
 
```bash
python -m venv sign
```
 
### 4. Activate the virtual environment
 
#### Windows
 
```bash
sign\Scripts\activate
```
 
### 5. Install dependencies
 
```bash
pip install -r requirements.txt
```
 
### 6. Run the application
 
```bash
streamlit run app.py
```
 
The application will open in your browser.
 
## 📷 How to Use
 
### 🖼️ Image Prediction
 
1. Open the **Image** page from the sidebar.
2. Select **Upload Image**.
3. Upload a clear image of an ASL hand sign.
4. Click **Predict**.
5. View the predicted letter and confidence score.
### 📸 Take a Picture
 
1. Select **Take Picture**.
2. Allow camera access.
3. Capture a clear hand sign.
4. Click **Predict**.
5. View the prediction and top-5 results.
### 📹 Webcam Prediction
 
1. Select **Webcam** from the sidebar.
2. Click **START**.
3. Allow browser camera access.
4. Show one ASL hand sign clearly.
5. Hold the sign steady.
6. The system stabilizes the detected landmarks before making a prediction.
7. The prediction remains locked until the hand is removed.
## ⚠️ Limitations
 
- The application detects **one hand at a time**.
- Hand landmarks may be less reliable when fingers overlap.
- Signs such as **M, N, Q, C, and X** can be more challenging for landmark detection.
- Prediction performance can be affected by lighting, background, hand position, and camera quality.
- Some predictions may be incorrect even when the confidence score is high.
## 📓 Training Notebook
 
The model development and training process is documented in:
 
```text
asl_ann.ipynb
```
 
The notebook contains the dataset processing, feature extraction, model training, evaluation, and hyperparameter tuning workflow.
 
## 👨‍💻 Author
 
**Dhulipala Prem Aditya**
 
- LinkedIn: https://www.linkedin.com/in/prem-aditya-dhulipala-627a43268
- GitHub: https://github.com/premaditya
## 📄 License
 
This project is intended for educational and portfolio purposes.