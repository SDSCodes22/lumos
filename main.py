# TODO
"""
This file should be recording camera video and voice and whenever the voice is finished it should send it to one
of the related models
"""
import cv2
import numpy as np
import os
import speech_recognition as sr

cap = cv2.VideoCapture(0)


# Function to encode the image
def _encode_image(self, image_path):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    encoded_image = base64.b64encode(image_data)
    return encoded_image.decode("utf-8")


def saveImg(frame):
    """
    Given an OpenCV Image (as an ndarray), it will convert to Base64 and save as a txt
    """
    if os.path.isfile("frame.txt"):
        os.remove("frame.txt")
    cv2.imwrite("frame.jpg", frame)
    txt = _encode_image("frame.jpg")
    with open("frame.txt", "w") as file:
        file.write(txt)


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
        # TODO: Send this data to relevant model

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
