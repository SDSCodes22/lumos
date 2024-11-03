import time
import os
import torch
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny",  # select checkpoint from https://huggingface.co/openai/whisper-large-v3#model-details
    torch_dtype=torch.float16,
    device="mps",  # or mps for Mac devices
    model_kwargs=(
        {"attn_implementation": "flash_attention_2"}
        if is_flash_attn_2_available()
        else {"attn_implementation": "sdpa"}
    ),
)
filename = os.path.join(os.path.dirname(__file__), "output.wav")
start_time = time.time()
outputs = pipe(
    filename,
    chunk_length_s=30,
    batch_size=24,
    return_timestamps=True,
)
print(outputs)
print(f"Took {time.time() - start_time} seconds!")
"""
chunk = 1024  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 1
fs = 44100  # Record at 44100 samples per second
seconds = 3
filename = "output.wav"

p = pyaudio.PyAudio()  # Create an interface to PortAudio


print("Ready for input! Press Space, then start talking for live transcription")
while True:
    if keyboard.is_pressed("space"):
        stream = p.open(
            format=sample_format,
            channels=channels,
            rate=fs,
            frames_per_buffer=chunk,
            input=True,
        )

        frames = []  # Initialize array to store frames
        isPressed = True
        print("Recording!")
        while isPressed:
            data = stream.read(chunk)
            frames.append(data)
            isPressed = keyboard.is_pressed("space")
        print("\nStopped Recording!")
        # Save the recorded data as a WAV file
        wf = wave.open(filename, "wb")
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b"".join(frames))
        wf.close()

        print("Saved temp wav file!\nPassing to Whisper!")
        outputs = pipe(
            filename,
            chunk_length_s=30,
            batch_size=24,
            return_timestamps=True,
        )
        print(f"OUTPUT: {outputs}")
"""
