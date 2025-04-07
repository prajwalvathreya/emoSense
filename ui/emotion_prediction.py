from os import pread
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

def predict_emotion(model, img):
    """
    Predict emotion from an image using the trained model

    Parameters:
    image_path (str): Path to the image file

    Returns:
    str: Predicted emotion label
    """

    emotion_labels = ['Anger', 'Disgust', 'Fear', 'Sadness', 'Neutral', 'Happy', 'Surprise']
    img = cv2.resize(img, (48, 48))
    img_arr = image.img_to_array(img)

    img_arr = np.expand_dims(img_arr, axis=0)

    img_arr = img_arr / 255.0

    predictions = model.predict(img_arr)
    print(predictions)

    predicted_emotion = emotion_labels[np.argmax(predictions)]
    confidence = max(predictions[0]) * 100

    return {
        "emotion": predicted_emotion,
        "confidence": f"{confidence:.2f}%",
        "all_probabilities" : {emotion_labels[i]: round(predictions[0][i], 2) for i in range(len(emotion_labels))}
    }
