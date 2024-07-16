# TODO
"""
This file should be recording camera video and voice and whenever the voice is finished it should send it to one
of the related models

TODO ON THE DAY
- Fix WorldBlob incorrectly classifying
- Try to get Object Detection working
- Fix classifier model
"""
import cv2
import numpy as np
import os
import speech_recognition as sr
from models.gpt.gpt_main import GPTInteracter
from models.scene_text_recognition.str_main import SceneTextRecognition
from models.facial_recognition.recognition_main import FacialRecognition_Lumos
from gtts import gTTS
from playsound import playsound
import threading
from tensorflow import keras
import tensorflow as tf
from transformers import BertTokenizer
from transformers import TFBertModel

cap = cv2.VideoCapture(0)
# Initialize
gem = GPTInteracter()
strrec = SceneTextRecognition()
facialRec = FacialRecognition_Lumos(gem)
model_path = os.path.join(os.path.dirname(__file__), "main_classification_model.h5")
main_classifier = tf.keras.models.load_model(model_path)
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")


def saveImg(frame):
    """
    Given an OpenCV Image (as an ndarray), it will convert to Base64 and save as a txt
    """
    if os.path.isfile("frame.txt"):
        os.remove("frame.txt")
    cv2.imwrite("frame.jpg", frame)


def play_waiting_sound():
    os.system("afplay placeholder.mp3")


# Given input text, this will prepare that text for the model
def prepare_data(input_text, tokenizer):
    token = tokenizer.encode_plus(
        input_text,
        max_length=256,
        truncation=True,
        padding="max_length",
        add_special_tokens=True,
        return_tensors="tf",
    )
    return {
        "input_ids": tf.cast(token.input_ids, tf.float64),
        "attention_mask": tf.cast(token.attention_mask, tf.float64),
    }


# Short-hand to process model output
def make_prediction(model, processed_data, classes=[0, 1, 2, 3]) -> int:
    probs = model.predict(processed_data)[0]
    return classes[np.argmax(probs)]


def classify_prompt(prompt: str) -> int:
    # Use classifier model to predict what type of question it is
    processed_prompt = prepare_data(prompt, tokenizer)
    model_case = make_prediction(main_classifier, processed_prompt)
    return model_case


def generate_audio(text: str) -> None:
    if os.path.isfile("response.mp3"):
        os.remove("response.mp3")
    sound = gTTS(text)
    sound.save("response.mp3")
    # Play with system
    os.system("afplay response.mp3")


# The speech recognition is done in a background thread to keep the main thread free for other tasks
# this is called from the background thread
def callback(recognizer, audio):
    # received audio data, now we'll recognize it using Google Speech Recognition
    try:
        # This is before .recognize_google to minimize delays
        FRAME = cap.read(0)[1]  #! Could be null
        # play a placeholder sound in a seperate thread
        thread = threading.Thread(target=play_waiting_sound)
        thread.start()
        SPEECH = recognizer.recognize_google(audio)

        print(f"SPEECH: {SPEECH}")
        print(f"debug: frame dtype = {FRAME.dtype}")
        saveImg(FRAME)

        #! Synchronous code
        print("Classifying prompt!")
        match classify_prompt(SPEECH):
            case 0:
                # LLM
                print("Sending to Gemini!")
                constant_starter = "This is pre-generated: You are an assistant who has been specially designed to help those who are either blind or visually blind. You are like a personal assistant whom these people can talk to. The image attached to this prompt is what this blind person is seeing in real time. The image is mirrored (left is right and vice-verse) In your response, do not use phrases like 'image', 'photo' or 'picture', however, assume that the user knows this already and instead use phrases like 'in front of you' or 'next to you'. You are like the user's eyes. This is the user prompt and the question you must answer: "
                txt = gem.get_response((constant_starter + SPEECH), FRAME)
                print(f"Received response: {txt}, sending to gTTS and playing audio")
                generate_audio(txt)

            case 1:
                # OCR
                print("Sending to OCR!")
                resp = strrec.generate_response(FRAME, SPEECH)
                if resp != None:
                    print(
                        f"Received response: {resp}, sending to gTTS and playing audio"
                    )
                    generate_audio(resp)

            case 2:
                # OCR
                print("Sending to PR!")
                resp = facialRec.generate_response(FRAME, SPEECH)
                if resp != None:
                    print(
                        f"Received response: {resp}, sending to gTTS and playing audio"
                    )
                    generate_audio(resp)
            case _:
                raise Exception(
                    "This case should never occur! Something terrible has happened"
                )

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
