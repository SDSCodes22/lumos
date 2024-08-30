import torch
import numpy as np
from scipy.io.wavfile import write
import time

# Function to load FastSpeech 2 model
def load_fastspeech2(model_path):
    model = torch.hub.load('pytorch/fairseq', 'fastspeech2', model_path)
    model.eval().cuda()
    return model

# Function to load HiFi-GAN vocoder
def load_hifigan(model_path):
    model = torch.hub.load('pytorch/hifi-gan', 'hifi-gan', model_path)
    model.eval().cuda()
    return model

# Function to synthesize speech
def synthesize_speech(text, fastspeech2, hifigan):
    with torch.no_grad():
        # Convert text to sequence
        input_text = torch.LongTensor([text_to_sequence(text)]).cuda()

        # Generate mel spectrogram using FastSpeech 2
        start_time = time.time()
        mel_output = fastspeech2(input_text)
        mel_gen_time = time.time() - start_time

        # Generate audio using HiFi-GAN vocoder
        start_time = time.time()
        audio_output = hifigan(mel_output)
        vocoder_gen_time = time.time() - start_time

    return audio_output, mel_gen_time, vocoder_gen_time

if __name__ == "__main__":
    # Load pre-trained models
    fastspeech2_path = 'path_to_fastspeech2_model.pth'
    hifigan_path = 'path_to_hifigan_model.pth'
    fastspeech2 = load_fastspeech2(fastspeech2_path)
    hifigan = load_hifigan(hifigan_path)

    # Text to synthesize
    text = "This is a test for low-latency TTS."

    # Synthesize speech and measure latency
    audio_output, mel_gen_time, vocoder_gen_time = synthesize_speech(text, fastspeech2, hifigan)

    # Calculate total latency
    total_latency = mel_gen_time + vocoder_gen_time
    print(f"Total Latency: {total_latency:.3f} seconds")

    # Save the output audio
    audio_output = audio_output.squeeze().cpu().numpy()
    write("output.wav", 22050, audio_output)
