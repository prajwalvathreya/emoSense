# import pyaudio
# import wave
# import datetime
# import os

# # Function to get a formatted timestamp
# def get_timestamp():
#     return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# # Take user input for the base filename
# base_filename = input("Enter a base filename: ")

# # Use current timestamp if no base filename is provided
# if not base_filename:
#     base_filename = get_timestamp()
# else:
#     base_filename = f"{base_filename}_{get_timestamp()}"

# # Define the directory to save the recordings
# recordings_dir = "recordings"

# # Create the recordings directory if it doesn't exist
# if not os.path.exists(recordings_dir):
#     os.makedirs(recordings_dir)

# # Define parameters for recording
# FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
# CHANNELS = 1              # Mono audio
# RATE = 44100              # Sample rate (44.1 kHz)
# CHUNK = 1024              # Buffer size per read
# RECORD_SECONDS = 5        # Duration of recording (in seconds)
# OUTPUT_FILENAME = os.path.join(recordings_dir, f"{base_filename}.wav") 

# # Initialize the audio stream
# p = pyaudio.PyAudio()

# # Open the audio stream
# stream = p.open(format=FORMAT,
#                 channels=CHANNELS,
#                 rate=RATE,
#                 input=True,
#                 frames_per_buffer=CHUNK)

# print("Recording...")

# # Record the audio
# frames = []
# for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
#     data = stream.read(CHUNK)
#     frames.append(data)

# # Stop and close the stream
# print("Recording finished.")
# stream.stop_stream()
# stream.close()
# p.terminate()

# # Save the recorded audio to a .wav file
# with wave.open(OUTPUT_FILENAME, 'wb') as wf:
#     wf.setnchannels(CHANNELS)
#     wf.setsampwidth(p.get_sample_size(FORMAT))
#     wf.setframerate(RATE)
#     wf.writeframes(b''.join(frames))

# print(f"Audio saved as {OUTPUT_FILENAME}")



import pyaudio
import numpy as np

# Function to record audio and return the audio data as an array
def record_audio(record_seconds=8):
    # Initialize PyAudio
    p = pyaudio.PyAudio()

    FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
    CHANNELS = 1              # Mono audio
    RATE = 44100              # Sample rate (44.1 kHz)
    CHUNK = 1024              # Buffer size per read

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    frames = []
    
    print("Recording...")
    for _ in range(0, int(RATE / CHUNK * record_seconds)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording finished.")
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Convert frames into a NumPy array
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
    
    return audio_data