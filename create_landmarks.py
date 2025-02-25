import cv2
import numpy as np
import mediapipe as mp
import os
import pandas as pd
import pickle
from tqdm import tqdm

# Initialize MediaPipe Face Landmarker
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

def extract_and_save_landmarks(dataset_path, output_path):
    """
    Extract facial landmarks from AffectNet images and save them
    Track failed face detections in a separate CSV
    
    Args:
        dataset_path: Path to AffectNet dataset
        output_path: Path to save landmarks and failed detection records
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Create dictionary to store emotion labels and landmarks
    data = {
        'image_path': [],
        'emotion_label': [],
        'landmarks': []
    }
    
    # Track failed face detections
    failed_detections = {
        'image_path': [],
        'emotion_label': [],
        'reason': []
    }
    
    # Iterate through dataset folders (assuming AffectNet structure)
    emotions = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt']
    
    for emotion_idx, emotion in enumerate(emotions):
        emotion_dir = os.path.join(dataset_path, emotion)
        if not os.path.exists(emotion_dir):
            continue
            
        image_files = [f for f in os.listdir(emotion_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        for img_file in tqdm(image_files, desc=f"Processing {emotion}"):
            img_path = os.path.join(emotion_dir, img_file)
            
            # Read image
            image = cv2.imread(img_path)
            if image is None:
                # Track failed image reading
                failed_detections['image_path'].append(img_path)
                failed_detections['emotion_label'].append(emotion_idx)
                failed_detections['reason'].append("Failed to read image file")
                continue
                
            # Convert to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process image with MediaPipe
            results = face_mesh.process(image_rgb)
            
            # Check if face detected
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                
                # Convert landmarks to numpy array (x, y, z) coordinates
                landmarks_array = np.array([
                    [lm.x, lm.y, lm.z] for lm in face_landmarks.landmark
                ])
                
                # Store data
                data['image_path'].append(img_path)
                data['emotion_label'].append(emotion_idx)
                data['landmarks'].append(landmarks_array)
            else:
                # Track failed face detection
                failed_detections['image_path'].append(img_path)
                failed_detections['emotion_label'].append(emotion_idx)
                failed_detections['reason'].append("No face detected")
    
    # Save landmarks data as pickle file
    with open(os.path.join(output_path, 'landmarks_data.pkl'), 'wb') as f:
        pickle.dump(data, f)
    
    # Save landmarks metadata as CSV
    landmarks_df = pd.DataFrame({
        'image_path': data['image_path'],
        'emotion_label': data['emotion_label'],
        'emotion_name': [emotions[idx] for idx in data['emotion_label']]
    })
    landmarks_df.to_csv(os.path.join(output_path, 'landmarks_metadata.csv'), index=False)
    
    # Save failed detections as CSV
    failed_df = pd.DataFrame({
        'image_path': failed_detections['image_path'],
        'emotion_label': failed_detections['emotion_label'],
        'emotion_name': [emotions[idx] for idx in failed_detections['emotion_label']],
        'reason': failed_detections['reason']
    })
    failed_df.to_csv(os.path.join(output_path, 'failed_detections.csv'), index=False)
    
    # Generate summary report
    print(f"Processing Summary:")
    print(f"Total images processed: {len(data['image_path']) + len(failed_detections['image_path'])}")
    print(f"Successful face detections: {len(data['image_path'])}")
    print(f"Failed face detections: {len(failed_detections['image_path'])}")
    
    # Report by emotion category
    print("\nSuccess rate by emotion category:")
    for emotion_idx, emotion in enumerate(emotions):
        successful = sum(1 for label in data['emotion_label'] if label == emotion_idx)
        failed = sum(1 for label in failed_detections['emotion_label'] if label == emotion_idx)
        total = successful + failed
        
        if total > 0:
            success_rate = (successful / total) * 100
            print(f"{emotion}: {successful}/{total} ({success_rate:.2f}%)")
    
    # Report by failure reason
    reasons = pd.Series(failed_detections['reason']).value_counts()
    print("\nFailure reasons:")
    for reason, count in reasons.items():
        print(f"{reason}: {count}")

# Example usage
if __name__ == "__main__":
    # Update these paths
    affectnet_path = "data\\3"
    output_path = "landmarks"
    
    extract_and_save_landmarks(affectnet_path, output_path)