import os
import numpy as np
import tensorflow as tf
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
 
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "final_ann_new.keras")
ENCODER_PATH = os.path.join(BASE_DIR, "..", "models", "label_encoder_new.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "..", "models", "scaler_new.pkl")
 
model = tf.keras.models.load_model(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
scaler = joblib.load(SCALER_PATH)

def prediction(features):

    if features is None:
        return None

    features = features.reshape(1, -1)

    features = scaler.transform(features)

    prob = model.predict(features, verbose = 0)

    top_5_indices = np.argsort(prob[0])[-5:][::-1]

    top_5_letters = encoder.inverse_transform(top_5_indices)

    top_5_confidence = prob[0][top_5_indices]

    top_5 = list(zip(top_5_letters, top_5_confidence))

    return top_5


