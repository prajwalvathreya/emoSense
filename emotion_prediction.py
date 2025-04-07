import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import itertools

def predict_emotion(model, image):
    """
    Predict emotion from an image using the trained model
    
    Parameters:
    image_path (str): Path to the image file
    
    Returns:
    str: Predicted emotion label
    """
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )

    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    # image = cv2.imread(image)
    # if image is None:
    #     return "Error: Could not read image"
        
    # # Resize image to match the training input size

    img_resized = cv2.resize(image, (48, 48))
    
    # Convert to RGB (MediaPipe works with RGB)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.GaussianBlur(img_rgb, (3, 3), cv2.BORDER_DEFAULT)
    
    # Process image with MediaPipe
    results = face_mesh.process(img_rgb)
    
    if not results.multi_face_landmarks:
        return "No face detected in the image"
    
    # Since we're using a CNN model, prepare the image for direct prediction
    # Normalize the resized image
    img_input = img_resized / 255.0
    
    # Expand dimensions to match the model's input shape (batch_size, height, width, channels)
    img_input = np.expand_dims(img_input, axis=0)
    
    # Make prediction
    prediction = model.predict(img_input)
    predicted_class_index = np.argmax(prediction)
    predicted_emotion = emotion_labels[predicted_class_index]
    confidence = prediction[0][predicted_class_index] * 100
    
    return {
        "emotion": predicted_emotion,
        "confidence": f"{confidence:.2f}%",
        "all_probabilities" : {emotion_labels[i]: round(prediction[0][i], 2) for i in range(len(emotion_labels))}
    }

# Example usage
if __name__ == "__main__":
    model = load_model('best_fer_model.h5')
    result = predict_emotion(model, "final_data/1/train/fearful/im5.png")
    print(result["emotion"])