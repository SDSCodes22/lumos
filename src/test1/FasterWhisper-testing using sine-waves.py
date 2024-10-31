import time
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
import matplotlib.pyplot as plt
import platform
import torch

# Define your device information
DEVICE_INFO = {
    "device_name": platform.node(),
    "device_type": platform.system(),
    "device_architecture": platform.machine(),
}

# Define different lengths of audio files in seconds for testing
audio_lengths = [10, 30, 60, 120, 300]  # Modify based on desired lengths (in seconds)

# Initialize the model
model = WhisperModel("tiny", device="cuda" if torch.cuda.is_available() else "cpu")

# Function to create a test audio file of specified length
def generate_sine_wave_audio(duration_seconds, sample_rate=16000):
    """Generates a sine wave audio file of given duration and sample rate."""
    t = np.linspace(0, duration_seconds, int(duration_seconds * sample_rate), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz sine wave
    return audio, sample_rate

# Measure transcription time for different audio lengths
results = []
for length in audio_lengths:
    # Generate or load an audio file with specified duration
    audio_data, sample_rate = generate_sine_wave_audio(length)

    # Write the generated audio to a temporary file
    temp_audio_file = f"temp_audio_{length}s.wav"
    sf.write(temp_audio_file, audio_data, sample_rate)

    # Time the transcription
    start_time = time.time()
    result = model.transcribe(temp_audio_file)
    transcription_time = time.time() - start_time

    # Log the result
    results.append((length, transcription_time))
    print(f"Audio Length: {length}s, Transcription Time: {transcription_time:.2f}s")

# Plot results
audio_lengths, transcription_times = zip(*results)
plt.plot(audio_lengths, transcription_times, marker='o')
plt.xlabel("Audio Length (seconds)")
plt.ylabel("Transcription Time (seconds)")
plt.title(f"FasterWhisper Speed Test on {DEVICE_INFO['device_name']} ({DEVICE_INFO['device_architecture']})")
plt.grid(True)
plt.show()

# Save results
with open("faster_whisper_speed_test_results.txt", "w") as f:
    f.write("Device Info:\n")
    for key, value in DEVICE_INFO.items():
        f.write(f"{key}: {value}\n")
    f.write("\nAudio Length (s) | Transcription Time (s)\n")
    f.write("-" * 30 + "\n") # type: ignore
    for length, transcription_time in results:
        f.write(f"{length:<16} | {transcription_time:.2f}\n")

print("Test completed and results saved.")
