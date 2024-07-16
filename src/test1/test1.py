import threading
import cv2
import pyaudio
import wave
import speech_recognition as sr
from transformers import pipeline
from gtts import gTTS
import os

# Initialize recognizer and NLP pipeline
recognizer = sr.Recognizer()
nlp_pipeline = pipeline('text-generation', model='gpt2')  # Make sure PyTorch is installed


# Function to capture audio
def capture_audio(frames, stream, p):
    for i in range(0, int(44100 / 1024 * 5)):  # 5 seconds of audio
        data = stream.read(1024)
        frames.append(data)


# Function to recognize speech from captured audio
def recognize_speech(frames, p):
    wf = wave.open("temp_audio.wav", 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(44100)
    wf.writeframes(b''.join(frames))
    wf.close()
    with sr.AudioFile("temp_audio.wav") as source:
        audio = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return "Could not understand the audio"
        except sr.RequestError:
            return "Could not request results; check your internet connection"


# Function to process video
def process_video(cap, stop_event):
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


# Function to generate response
def generate_response(input_text):
    response = nlp_pipeline(input_text, max_length=50)
    return response[0]['generated_text']


# Function to convert text to speech
def text_to_speech(text):
    tts = gTTS(text)
    tts.save("response.mp3")
    os.system("start response.mp3")  # Use "afplay" for MacOS or "xdg-open" for Linux


# Main function
def main():
    # Initialize video and audio capture
    cap = cv2.VideoCapture(0)
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)

    stop_event = threading.Event()

    while True:
        frames = []

        # Capture audio in a separate thread
        audio_thread = threading.Thread(target=capture_audio, args=(frames, stream, p))
        audio_thread.start()

        # Process video in a separate thread
        video_thread = threading.Thread(target=process_video, args=(cap, stop_event))
        video_thread.start()

        # Wait for audio capture to complete
        audio_thread.join()

        # Recognize speech and generate response
        spoken_text = recognize_speech(frames, p)
        print("Recognized Text:", spoken_text)
        response = generate_response(spoken_text)
        print("Generated Response:", response)

        # Convert response to speech
        text_to_speech(response)

        if spoken_text.lower() == "exit":
            stop_event.set()
            video_thread.join()
            break

    # Release resources
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Ensure video capture is released and windows are closed
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
