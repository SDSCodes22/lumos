import threading
#Threading allows you to have different parts of your process run concurrently
import cv2
#Allowing access to OpenCV's functions for computer vision and image processing
import pyaudio
#for audio playback
import wave
import speech_recognition as sr
#using the SpeechRecognition library
from transformers import pipeline
#enables the table function to return rows faster
from gtts import gTTS
#Google Text-to-Speech
import os
#to interact with the underlying operating system

# Initialize recognizer and NLP pipeline
recognizer = sr.Recognizer()
nlp_pipeline = pipeline('text-generation', model='gpt3')  # Make sure PyTorch is installed


# Function to capture audio
def capture_audio(frames, stream, p):
    for i in range(0, int(44100 / 1024 * 5)):  # 5 seconds of audio
        data = stream.read(1024)
        frames.append(data)


# Function to recognize speech from captured audio
def recognize_speech(frames, p):
    # Create a WAV file to store the audio data
    wf = wave.open("temp_audio.wav", 'wb')
    wf.setnchannels(1)  # Set the number of audio channels to 1 (mono)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))  # Set sample width to match the PyAudio format
    wf.setframerate(44100)  # Set the frame rate to 44.1 kHz (standard for audio files)
    wf.writeframes(b''.join(frames))  # Write the captured audio frames to the WAV file
    wf.close()  # Close the WAV file

    # Use the SpeechRecognition library to transcribe the audio
    with sr.AudioFile("temp_audio.wav") as source:
        audio = recognizer.record(source)  # Record the audio from the WAV file
        try:
            # Attempt to recognize the speech using the Google Web Speech API
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            # Handle cases where the speech was not understood
            return "Could not understand the audio"
        except sr.RequestError:
            # Handle cases where the API request failed (e.g., no internet connection)
            return "Could not request results; check your internet connection"


# Function to process video
def process_video(cap, stop_event):
    # Loop until the stop event is set
    while not stop_event.is_set():
        ret, frame = cap.read()  # Capture frame from the video stream
        if not ret:
            break  # Break the loop if frame is not captured
        cv2.imshow('Video', frame)  # Display the frame in a window
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit loop if 'q' key is pressed
            break
    cap.release()  # Release the video capture object
    cv2.destroyAllWindows()  # Close all OpenCV windows


# Function to generate response
def generate_response(input_text):
    response = nlp_pipeline(input_text, max_length=50)  # Generate text response using NLP pipeline
    return response[0]['generated_text']  # Return the generated text


# Function to convert text to speech
def text_to_speech(text):
    tts = gTTS(text)  # Convert text to speech using gTTS
    tts.save("response.mp3")  # Save the speech as an MP3 file
    os.system("start response.mp3")  # Play the MP3 file (use "afplay" for MacOS or "xdg-open" for Linux)


# Main function
def main():
    # Initialize video and audio capture
    cap = cv2.VideoCapture(0)  # Open the default camera
    p = pyaudio.PyAudio()  # Initialize PyAudio
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)  # Open audio stream

    stop_event = threading.Event()  # Create an event to signal stopping

    while True:
        frames = []  # List to store audio frames

        # Capture audio in a separate thread
        audio_thread = threading.Thread(target=capture_audio, args=(frames, stream, p))
        audio_thread.start()  # Start the audio capture thread

        # Process video in a separate thread
        video_thread = threading.Thread(target=process_video, args=(cap, stop_event))
        video_thread.start()  # Start the video processing thread

        # Wait for audio capture to complete
        audio_thread.join()  # Wait for the audio thread to finish

        # Recognize speech and generate response
        spoken_text = recognize_speech(frames, p)  # Convert audio frames to text
        print("Recognized Text:", spoken_text)  # Print the recognized text
        response = generate_response(spoken_text)  # Generate a response based on the recognized text
        print("Generated Response:", response)  # Print the generated response

        # Convert response to speech
        text_to_speech(response)  # Convert the response text to speech

        if spoken_text.lower() == "exit":  # Check if the recognized text is "exit"
            stop_event.set()  # Set the stop event to stop the video processing
            video_thread.join()  # Wait for the video thread to finish
            break  # Exit the loop

    # Release resources
    stream.stop_stream()  # Stop the audio stream
    stream.close()  # Close the audio stream
    p.terminate()  # Terminate the PyAudio object

    # Ensure video capture is released and windows are closed
    cap.release()  # Release the video capture object
    cv2.destroyAllWindows()  # Close all OpenCV windows


if __name__ == "__main__":
    main()  # Execute the main function

