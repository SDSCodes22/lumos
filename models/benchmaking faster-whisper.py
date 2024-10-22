import time
from faster_whisper import WhisperModel # type: ignore

# Load the Whisper model
model = WhisperModel("medium", device="cuda", compute_type="float16")

# Load a test audio file (replace with your own)
audio_path = "test_audio.wav"

# Start timer
start_time = time.time()

# Transcribe the audio
segments, info = model.transcribe(audio_path)

# End timer
end_time = time.time()

# Output timing and transcription info
print(f"Model size: {info['model_size']}")
print(f"Language: {info['language']}")
print(f"Transcription time: {end_time - start_time:.2f} seconds")

# Print segments
for segment in segments:
    print(f"[{segment.start} - {segment.end}]: {segment.text}")
