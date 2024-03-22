# TODO
"""
This file should be recording camera video and voice and whenever the voice is finished it should send it to one
of the related models
"""
import cv2
import numpy as np
import os
import speech_recognition as sr
from models.gpt.gpt_main import GPTInteracter
from gtts import gTTS
from playsound import playsound
import threading

cap = cv2.VideoCapture(0)
# Initialize
gem = GPTInteracter()


def saveImg(frame):
    """
    Given an OpenCV Image (as an ndarray), it will convert to Base64 and save as a txt
    """
    if os.path.isfile("frame.txt"):
        os.remove("frame.txt")
    cv2.imwrite("frame.jpg", frame)


def generate_audio(txt: str) -> None:
    print("Generating audio...")
    tts = gTTS(txt)
    print("Audio generated successfully!")
    # Prevent override
    if os.path.isfile("response.mp3"):
        os.remove("response.mp3")
    print("Saving audio from gTTS as mp3 file...")
    tts.save("response.mp3")
    print("\tDone!\n\nPlaying response to user!")
    playsound("response.mp3")


def play_waiting_sound():
    playsound("placeholder.mp3")


# The speech recognition is done in a background thread to keep the main thread free for other tasks
# this is called from the background thread
def callback(recognizer, audio):
    # received audio data, now we'll recognize it using Google Speech Recognition
    try:
        # This is before .recognize_google to minimize delays
        FRAME = cap.read(0)[1]  #! Could be null
        # for testing purposes, we're just using the default API key
        # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
        # instead of `r.recognize_google(audio)`
        print(
            "Google Speech Recognition thinks you said "
            + recognizer.recognize_google(audio)
        )
        SPEECH = recognizer.recognize_google(audio)

        print(f"SPEECH: {SPEECH}")
        print(f"debug: frame dtype = {FRAME.dtype}")
        saveImg(FRAME)
        # play a placeholder sound in a seperate thread
        thread = threading.Thread(target=play_waiting_sound)
        thread.start()
        # TODO: Send this data to relevant model
        # ? For now, we're just going to send this to Gemini-1.0-Vision
        #! Synchronous code
        print("Sending to Gemini!")
        constant_starter = "This is pre-generated: You are an assistant who has been specially designed to help those who are either blind or visually blind. You are like a personal assistant whom these people can talk to. The image attached to this prompt is what this blind person is seeing in real time. In your response, do not use phrases like 'image', 'photo' or 'picture' however, assume that the user knows this already and instead use phrases like 'in front of you' or 'next to you'. This is the user prompt and the question you must answer: "
        txt = gem.get_response((constant_starter + SPEECH))
        print(f"Received response: {txt}, sending to gTTS and playing audio")
        generate_audio(txt)

    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(
            "Could not request results from Google Speech Recognition service; {0}".format(
                e
            )
        )


r = sr.Recognizer()
m = sr.Microphone()
with m as source:
    r.adjust_for_ambient_noise(
        source
    )  # we only need to calibrate once, before we start listening

# start listening in the background (note that we don't have to do this inside a `with` statement)
stop_listening = r.listen_in_background(m, callback)
# `stop_listening` is now a function that, when called, stops background listening

while True:
    ret, frame = cap.read(0)

    # this shows the video to the screen
    cv2.imshow("frame", frame)

    # Code to save the current frame as jpg
    if cv2.waitKey(1) == ord("s"):
        saveImg(frame)

    # Exit
    if cv2.waitKey(1) == ord("q"):
        stop_listening(wait_for_stop=False)  # Stop listening aswell
        break
