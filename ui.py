import streamlit as st
import time
import numpy as np
import cv2

# Configure the page
st.set_page_config(page_title="Speaker & Listener Emotion Detector", layout="wide")
st.title("Speaker & Listener Emotion Detector")

# Initialize session state variables if not already set.
if "recording" not in st.session_state:
    st.session_state.recording = False
if "start_time" not in st.session_state:
    st.session_state.start_time = None

# Layout: two columns (left: controls & info, right: video feed)
col1, col2 = st.columns([1, 2])

with col1:
    # Recording button: toggles state
    if st.session_state.recording:
        if st.button("Stop Recording"):
            st.session_state.recording = False
            st.session_state.start_time = None
    else:
        if st.button("Start Recording"):
            st.session_state.recording = True
            st.session_state.start_time = time.time()
    
    # Display the recording duration or a message if not recording.
    if st.session_state.recording:
        duration = int(time.time() - st.session_state.start_time)
        st.write(f"**Recording Duration:** {duration} seconds")
    else:
        st.write("Not recording")
    
    st.markdown("---")
    st.subheader("Emotions")
    st.write("**Speaker Emotion:** Placeholder")
    st.write("**Listener Emotion:** Placeholder")

with col2:
    st.subheader("Live Webcam Feed")
    # Create a dummy image as a placeholder for the live webcam feed.
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(dummy_frame, "Webcam Feed Placeholder", (50, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    st.image(dummy_frame, channels="BGR")
    
    st.markdown("---")
    st.subheader("Last Captured Image (every 10 sec)")
    # For the captured image placeholder, simply display the same dummy frame.
    st.image(dummy_frame, channels="BGR")
