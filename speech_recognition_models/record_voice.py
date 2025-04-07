import pyaudio
import numpy as np

# Function to record audio and return the audio data as an array
def record_audio(record_seconds=9):
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
