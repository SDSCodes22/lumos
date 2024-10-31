import time
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
import matplotlib.pyplot as plt
import platform
import torch
from moviepy.editor import AudioFileClip
import yt_dlp
import sys
import os

# Define your device information
DEVICE_INFO = {
    "device_name": platform.node(),
    "device_type": platform.system(),
    "device_architecture": platform.machine(),
}

# Define different lengths of audio files in seconds for testing
audio_lengths = [2, 4, 6, 8, 10]  # Modify based on desired lengths (in seconds)

# Initialize the Whisper model
model = WhisperModel("tiny", device="cuda" if torch.cuda.is_available() else "cpu")

# Function to download YouTube audio using yt-dlp
def download_youtube_audio(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'youtube_audio.mp4',
        'noplaylist': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return "youtube_audio.mp4"
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

# Convert downloaded audio to WAV format
def convert_to_wav(input_path, output_path="youtube_audio.wav"):
    clip = AudioFileClip(input_path)
    clip.write_audiofile(output_path, fps=16000, nbytes=2)  # 16 kHz, 16-bit audio
    clip.close()
    return output_path

# Extract an audio segment of specified length
def extract_audio_segment(audio_path, start_time, duration):
    """Extracts a segment of given duration from the audio at start_time."""
    clip = AudioFileClip(audio_path)
    clip_duration = clip.duration
    end_time = start_time + duration
    # Ensure end time does not exceed the clip's duration
    if end_time > clip_duration:
        end_time = clip_duration
    # Extract the audio segment and convert to numpy array
    segment = clip.subclip(start_time, end_time)
    if segment.duration > 0:
        audio_data = segment.to_soundarray(fps=16000, nbytes=2)[:, 0]  # Mono channel
    else:
        audio_data = np.array([])  # Return empty array if segment is invalid
    segment.close()
    return audio_data, 16000

# Download and convert the audio
youtube_url = "https://www.youtube.com/watch?v=jRWR0Ob6mLI"  # Replace with your video URL
audio_path_mp4 = download_youtube_audio(youtube_url)
audio_path = convert_to_wav(audio_path_mp4)

# Measure transcription time for different audio lengths
results = []
for length in audio_lengths:
    # Extract an audio segment of the specified duration
    audio_data, sample_rate = extract_audio_segment(audio_path, 0, length)

    if audio_data.size > 0:  # Only proceed if audio_data is not empty
        # Write the extracted audio to a temporary file
        temp_audio_file = f"temp_audio_{length}s.wav"
        sf.write(temp_audio_file, audio_data, sample_rate)

        # Time the transcription
        start_time = time.time()
        result = model.transcribe(temp_audio_file)
        transcription_time = time.time() - start_time

        # Log the result
        results.append((length, transcription_time))
        print(f"Audio Length: {length}s, Transcription Time: {transcription_time:.2f}s")
    else:
        print(f"Skipping length {length}s - audio segment was empty or invalid.")

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
    f.write("-" * 30 + "\n")
    for length, transcription_time in results:
        f.write(f"{length:<16} | {transcription_time:.2f}\n")

# Clean up
if os.path.exists(audio_path_mp4):
    os.remove(audio_path_mp4)
if os.path.exists(audio_path):
    os.remove(audio_path)

print("Test completed and results saved.")
