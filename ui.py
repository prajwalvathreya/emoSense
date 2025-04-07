import streamlit as st
import cv2
import time
import threading
import numpy as np
from PIL import Image
from emotion_prediction import predict_emotion  
from tensorflow.keras.models import load_model

# Set page configuration
st.set_page_config(
    page_title="Dual Emotion Recognition System",
    page_icon="😀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .emotion-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 10px 24px;
        margin: 10px 0;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .emotion-result {
        font-size: 24px;
        font-weight: bold;
        margin: 10px 0;
    }
    .header {
        text-align: center;
        font-family: 'Arial Black', Gadget, sans-serif;
        color: #333;
        margin-bottom: 30px;
    }
    .confidence-bar {
        height: 20px;
        border-radius: 5px;
        margin-bottom: 5px;
    }
    .status-text {
        font-style: italic;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown("<h1 class='header'>Dual Emotion Recognition System</h1>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; margin-bottom: 30px;'>
This application analyzes emotions from both facial expressions and voice.
Capture is performed every 10 seconds when started.
</div>
""", unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'start_time' not in st.session_state:
    st.session_state.start_time = 0
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'last_image' not in st.session_state:
    st.session_state.last_image = None
if 'facial_emotion' not in st.session_state:
    st.session_state.facial_emotion = None
if 'voice_emotion' not in st.session_state:
    st.session_state.voice_emotion = None
if 'capture_count' not in st.session_state:
    st.session_state.capture_count = 0
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False

# Placeholder function for facial emotion recognition
def analyze_facial_emotion(image):
    """
    Placeholder function for facial emotion recognition.
    Replace with your actual implementation.
    
    Args:
        image: The image to analyze
        
    Returns:
        dict: A dictionary with emotion labels as keys and confidence scores as values
    """
    model = load_model('best_fer_model.h5')
    result = predict_emotion(model, image)

    if isinstance(result, str):
        return None

    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print(result)

    return result["all_probabilities"]

# Placeholder function for voice emotion recognition
def record_and_analyze_voice():
    """
    Placeholder function for voice recording and emotion analysis.
    Replace with your actual implementation.
    
    Returns:
        dict: A dictionary with emotion labels as keys and confidence scores as values
    """
    # Simulate recording delay
    time.sleep(3)
    
    # Placeholder: Return random emotions with confidence scores
    emotions = {
        "Happy": 0.1,
        "Sad": 0.05,
        "Angry": 0.65,
        "Surprised": 0.05,
        "Neutral": 0.15
    }
    return emotions

def stop_recording():
    st.session_state.is_recording = False
    st.session_state.is_running = False

def start_analysis():
    if not st.session_state.is_running:
        st.session_state.is_running = True
        st.session_state.start_time = time.time()
        st.session_state.is_recording = True

# Create two columns for the main layout
col1, col2 = st.columns([3, 2])

# Left column for video feed and controls
with col1:
    st.markdown("<div class='emotion-card'>", unsafe_allow_html=True)
    st.markdown("### Live Camera Feed")
    
    # Placeholder for the webcam feed
    video_placeholder = st.empty()
    
    # Controls
    control_col1, control_col2 = st.columns(2)
    
    with control_col1:
        if st.button("Start Analysis" if not st.session_state.is_running else "Analysis Running..."):
            start_analysis()
    
    with control_col2:
        if st.button("Stop"):
            stop_recording()
    
    # Status indicator
    if st.session_state.is_running:
        elapsed = time.time() - st.session_state.start_time
        next_capture = max(0, 10 - (elapsed % 10))
        st.markdown(f"<p class='status-text'>Analysis active. Next capture in {next_capture:.1f} seconds</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='status-text'>System idle. Press Start to begin analysis.</p>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Last captured image section
    # st.markdown("<div class='emotion-card'>", unsafe_allow_html=True)
    # st.markdown("### Last Captured Image")
    # last_image_placeholder = st.empty()
    # capture_info = st.empty()
    # st.markdown("</div>", unsafe_allow_html=True)

# Right column for emotion results
with col2:
    # Facial emotion results
    st.markdown("<div class='emotion-card'>", unsafe_allow_html=True)
    st.markdown("### Facial Emotion Analysis")
    facial_result_placeholder = st.empty()

    print("=============================")
    print(st.session_state.facial_emotion)
    
    # Display facial emotion results
    if st.session_state.facial_emotion:
        facial_emotions = st.session_state.facial_emotion
        dominant_emotion = max(facial_emotions, key=facial_emotions.get)
        facial_result_placeholder.markdown(f"<p class='emotion-result'>Dominant: {dominant_emotion}</p>", unsafe_allow_html=True)
        
        # Sort emotions by confidence and get top 3
        top_emotions = dict(sorted(facial_emotions.items(), key=lambda x: x[1], reverse=True)[:3])
        
        # Display confidence bars for top 3 emotions
        for emotion, confidence in top_emotions.items():
            st.markdown(f"{emotion} ({confidence:.2f})")
            st.markdown(
                f"""<div class="confidence-bar" style="width: {int(confidence*100)}%; 
                background-color: {'#4CAF50' if emotion == dominant_emotion else '#ddd'}"></div>""", 
                unsafe_allow_html=True
            )
    else:
        facial_result_placeholder.info("No facial emotion data yet")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Voice emotion results
    st.markdown("<div class='emotion-card'>", unsafe_allow_html=True)
    st.markdown("### Voice Emotion Analysis")
    voice_result_placeholder = st.empty()
    
    # Display voice emotion results
    if st.session_state.voice_emotion:
        voice_emotions = st.session_state.voice_emotion
        dominant_emotion = max(voice_emotions, key=voice_emotions.get)
        voice_result_placeholder.markdown(f"<p class='emotion-result'>Dominant: {dominant_emotion}</p>", unsafe_allow_html=True)
        
        # Sort emotions by confidence and get top 3
        top_emotions = dict(sorted(voice_emotions.items(), key=lambda x: x[1], reverse=True)[:3])
        
        # Display confidence bars for top 3 emotions
        for emotion, confidence in top_emotions.items():
            st.markdown(f"{emotion} ({confidence:.2f})")
            st.markdown(
                f"""<div class="confidence-bar" style="width: {int(confidence*100)}%; 
                background-color: {'#1E88E5' if emotion == dominant_emotion else '#ddd'}"></div>""", 
                unsafe_allow_html=True
            )
    else:
        voice_result_placeholder.info("No voice emotion data yet")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Combined insight
    if st.session_state.facial_emotion and st.session_state.voice_emotion:
        st.markdown("<div class='emotion-card'>", unsafe_allow_html=True)
        st.markdown("### Emotional Alignment")
        
        facial_dominant = max(st.session_state.facial_emotion, key=st.session_state.facial_emotion.get)
        voice_dominant = max(st.session_state.voice_emotion, key=st.session_state.voice_emotion.get)
        
        if facial_dominant == voice_dominant:
            st.success(f"Match: Both face and voice indicate {facial_dominant}")
        else:
            st.warning(f"Mismatch: Face shows {facial_dominant} while voice indicates {voice_dominant}")
        
        st.markdown("</div>", unsafe_allow_html=True)

# Main app logic
cap = cv2.VideoCapture(0)

# Webcam processing loop
while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to access webcam")
        break
    
    # Convert frame from BGR to RGB for display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Display current frame
    video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
    
    # If analysis is running and it's time for capture
    if st.session_state.is_running:
        elapsed = time.time() - st.session_state.start_time
        
        # Check if 10 seconds have passed for a new capture
        if elapsed % 10 < 0.5 and (st.session_state.capture_count == 0 or elapsed // 10 > st.session_state.capture_count):
            # Update the capture count
            st.session_state.capture_count = elapsed // 10
            
            # Store the captured image
            st.session_state.last_image = frame_rgb.copy()
            
            # Analyze facial emotion
            st.session_state.facial_emotion = analyze_facial_emotion(frame_rgb)
            
            # Analyze voice emotion at the same time
            st.session_state.voice_emotion = record_and_analyze_voice()

            st.rerun()
            
            # Display last captured image
            # last_image_placeholder.image(st.session_state.last_image, channels="RGB", use_container_width=True)
            # capture_info.info(f"Analysis performed at {time.strftime('%H:%M:%S')}")
    
    # Short delay to reduce CPU usage
    time.sleep(0.1)
    
    # Break the loop if the app is rerun
    if not st.session_state.is_running and 'rerun_requested' in st.session_state:
        break

# Release webcam on app close
cap.release()