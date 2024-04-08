#! Importing these libraries takes a really long time, idk why
from loguru import logger as log

log.info("Importing all libraries")
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

log.info("Imported MediaPipe")
import os
import cv2
import numpy as np
import numpy.typing as npt
import math
import multiprocessing
import pandas as pd
from deepface import DeepFace

# from models.facial_recognition.gpt_main import GPTInteracter
from tensorflow import keras
import tensorflow as tf
from transformers import BertTokenizer
from transformers import TFBertModel

log.info("Imported tensorflow and BERT Tokenizer")
import speech_recognition as sr
from textblob import TextBlob
from gtts import gTTS
from random import randint
import spacy
from spacy import displacy

log.info("Done!")


class FacialRecognition_Lumos:
    def __init__(self, GPTInteractor):
        self.gpt = GPTInteractor
        model_path = os.path.join(
            os.path.dirname(__file__), "classifier", "classification_model.h5"
        )
        self.classifier_model = tf.keras.models.load_model(model_path)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        self.NER = spacy.load("en_core_web_sm")
        log.info("Finished initialization, ready for usage!")

    def generate_response(self, prompt: str, frame: npt.NDArray) -> str:
        """
        Given a frame image (as ndarray) and text, it will generate a response to any question
        along the lines of:
            0 - Who is here?
            1 - Is x here?
            2 - How many people are here?
            3 - Is there anybody here I don't know?
            *numbers are the model's representation
        Returns:
            -String resposne to the question
            -Returns None if detected as question type 3, as this class will handle it.
        """

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

        log.info("Classifier classifying prompt into a category...")
        # Use classifier model to predict what type of question it is
        processed_prompt = prepare_data(prompt, self.tokenizer)
        model_case = make_prediction(self.classifier_model, processed_prompt)
        log.info(f"Classifier classifyed prompt into model case {model_case}")

        # Get list of people's names recognised in image
        log.info("Finding faces in frame...")
        people: tuple[npt.NDArray] = self._find_faces_in_frame(frame)[0]
        log.info(f"{len(people)} have been identified in this frame.")
        log.info("Classifying Faces...")
        names = self._classify_faces(people)

        match model_case:
            # Case 0
            case 0:
                # remove unknown
                names.remove("???")
                match len(names):
                    case 0:
                        return "Nobody is here."
                    case 1:
                        return f"Only {names[0]} is here."
                    case _:
                        output = ""
                        for i, x in enumerate(names):
                            output += (
                                (x + ", ") if i != len(names) - 1 else ("and " + x)
                            )
                        return output + " are here."
            # Case 1
            case 1:
                # In the prompt, we need to indentify who is the name
                name = None
                named_entities = self.NER(prompt)
                log.info(f"Named entities: {named_entities}")
                for x in named_entities.ents:
                    if x.label_ == "PERSON":
                        name = x
                        break
                if name == None:
                    return "No name was detected in your question, sorry."
                if self.find_rect_of_x(
                    frame, name.text.split(" ")[0]
                ):  # Splitting because we don't need last names in this case
                    return f"Yes, {name} is here."
                else:
                    return f"No, {name} is not here."

            # Case 2
            case 2:
                match len(names):
                    case 0:
                        return "Nobody is here."
                    case 1:
                        return "One person is here."
                    case _:
                        return f"{len(names)} people are here."

            # Case 3
            case 3:
                gemini_output = []
                for i, x in enumerate(names):
                    if x != "???":
                        gemini_output.append("_")
                        continue
                    text = "The image attached displays a person's face. Your task is to describe this person to the best of your ability, staying concise. Your response should be short, and start with the words 'This person has'. Please generate a description of this person."
                    gemini_output.append(self.gpt.get_response(text, people[i]))
                log.info(f"Gemini finished processing faces. Output: {gemini_output}")
                count = [
                    x for x in gemini_output if x != "_"
                ]  # I don't want it to be in place so that's why not .remove()
                log.info("Telling user how many people it does not recognise.")
                match len(count):
                    case 0:
                        self.generate_audio(f"I can recognise everybody here")
                    case 1:
                        self.generate_audio(f"I can't recognise one person here.")
                    case _:
                        self.generate_audio(
                            f"I can't recognise {len(count)} people here."
                        )

                for i, x in enumerate(gemini_output):
                    if x != "_":
                        self.handle_adding_face(people[i], x)
        return None

    def generate_audio(self, txt: str) -> None:
        print("Generating audio...")
        tts = gTTS(txt)
        print("Audio generated successfully!")
        # Prevent override
        if os.path.isfile("response.mp3"):
            os.remove("response.mp3")
        print("Saving audio from gTTS as mp3 file...")
        tts.save("response.mp3")
        print("\tDone!\n\nPlaying response to user!")
        os.system("afplay response.mp3")

    def handle_adding_face(self, frame: npt.NDArray, gemini_output: str) -> None:
        """
        Synchronous Front-End Function that handles user control flow to add a face
        to the database
        """
        log.info("Handling adding face.")
        self.generate_audio(
            f"I cannot recognise a person. {gemini_output}. Do you recognise them?"
        )

        def _listen_to_mic() -> str:
            # obtain audio from the microphone
            r = sr.Recognizer()
            with sr.Microphone() as source:
                audio = r.listen(source)

            # recognize speech using Sphinx
            try:
                output = r.recognize_sphinx(audio)
                log.info(f"Recognised:{output}")
                return output
            except sr.UnknownValueError:
                print("Sphinx could not understand audio!")
                return "_"
            except sr.RequestError as e:
                print("Sphinx error; {0}".format(e))
                return "_"

        response = _listen_to_mic()
        while response == "_":
            self.generate_audio("I did not hear that. Try again!")
            _listen_to_mic()

        match response == "Yes":
            case True:
                self.generate_audio(
                    "Perfect, Please say only their first name now and I will them to my knowledge."
                )
                resp = _listen_to_mic()

                while resp == "_":
                    self.generate_audio("I did not hear that. Try again!")
                    _listen_to_mic()

                cv2.imwrite(
                    os.path.join(
                        os.path.dirname(__file__),
                        "faces",
                        f"{resp.strip().lower()}_{randint(0,1000000)}.jpg",
                    ),
                    frame,
                )

            case False:
                self.generate_audio("That's good to know, I'll remember that!")
                # We should still save this person's face as unknown so we don't ask the user ever again
                cv2.imwrite(
                    os.path.join(
                        os.path.dirname(__file__),
                        "faces",
                        f"unknown_{randint(0,1000000)}.jpg",
                    ),
                    frame,
                )  #! There is approx. realistically 1 in a 100,000 chance this line fails

    # Copyright Soumyadeep Saha 2023 Under the MIT License
    def find_rect_of_x(self, frame: npt.NDArray, name: str) -> bool:
        """
        Given a string which is the name of the desired person, it will attempt to find the
        person in the given frame/image.

        Returns:
            Boolean of whether x is in the image
        """
        # We can just use DeepFace.find()
        # Parse the name into an "identity"
        name_path = name.replace(" ", "_")  # Joe Biden -> Joe_Biden
        name_path = name_path.lower() + ".jpg"  # Joe_Biden -> joe_biden.jpg
        name_path = os.path.join(
            "faces/", name_path
        )  # joe_biden.jpg -> faces/joe_biden.jpg OR faces\joe_biden.jpg
        # make the name path absolute
        absolute_name_path = os.path.join(os.path.dirname(__file__), name_path)

        #! Takes ~1 second to run
        try:
            result = DeepFace.verify(absolute_name_path, frame)
            return result["verified"]
        except Exception as e:
            # This probably occurred because the face is not even in the database, so it is impossible
            # for it to be a match, so return false
            return False

    # Worker function to find the identity of given face
    def find_face(self, face):

        try:
            db_path = os.path.join(os.path.dirname(__file__), "faces/")

            models = [  # TODO: In the line DeepFace.find, change the model index to choose any model from this list to compare the performance
                "VGG-Face",
                "Facenet",
                "Facenet512",
                "OpenFace",
                "DeepFace",
                "DeepID",
                "ArcFace",
                "Dlib",
                "SFace",
                "GhostFaceNet",
            ]

            df = DeepFace.find(img_path=face, db_path=db_path, model_name=models[0])[
                0
            ]  # Index 0 because it'll be a list with many faces
            # Check if the df is empty, because then that means there was no match
            if df.empty:
                return "???"
            # Now we should have a df with important columns identity and distance. We need to find the identity with the minimum distance
            min_distance_row = df[df["distance"] == df["distance"].min()]
            identity: str = min_distance_row["identity"]

            # If the minimum distance still didn't meet the threshold, it means the person was not identified
            if min_distance_row["distance"] < distance_threshold:
                # Now we have an identity looking like eg. "faces/joe_biden.jpg" so we need to parse it
                name = identity.split("/")[1].split(".")[0]
                name = name.replace("_", " ")

                return name
            else:
                return "???"
        except Exception as e:
            print(
                f"There was an error trying to find the face. Here is the exact error: {e}"
            )
            return "???"

    def _classify_faces(
        self, faces: list, model_index: int = 0, distance_threshold: float = 0.5
    ) -> list[str]:
        """
        Given a list of cropped faces (as 3D ndarrays), this will classify them if they have already been
        added to the folder /faces/

        If they are not recognised, ??? will be returned
        """

        # Loop through each face
        log.info("Looping through faces...")
        output = []
        for i, x in enumerate(faces):
            output.append(self.find_face(x))
        log.info(f"Done!, output: {output}")
        return output

    def _find_faces_in_frame(self, frame) -> tuple:
        """
        Given an image as an ndarray, this will identify all the human faces and return
        a list of all these cropped faces, where each element is a face
        """
        # ? Initialize library
        model_path = os.path.join(
            os.path.dirname(__file__), "blaze_face_short_range.tflite"
        )
        BaseOptions = mp.tasks.BaseOptions
        FaceDetector = mp.tasks.vision.FaceDetector
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Create a face detector instance with the image mode:
        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            min_detection_confidence=0.5,
        )

        bounds = []  # Will store the bounding box of each face for later

        detector = FaceDetector.create_from_options(options)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # Get results
        detector_result = detector.detect(mp_img)
        # Parse Results into List
        for detection in detector_result.detections:
            bounds.append(detection.bounding_box)

        # Crop image for item in list
        def _get_rect_points(bbox, padding=100) -> dict:
            output = dict()
            output["min_x"] = bbox.origin_x
            output["max_x"] = bbox.origin_x + bbox.width
            output["min_y"] = bbox.origin_y
            output["max_y"] = bbox.origin_y + bbox.height

            return output

        output = []
        for i in bounds:
            # TODO: Modify the padding arg here to change how tight the faces are
            p = _get_rect_points(i)

            output.append(
                frame[
                    int(p["min_y"]) - 100 : int(p["max_y"]) + 100,
                    int(p["min_x"]) - 100 : int(p["max_x"]) + 100,
                ]
            )

        return (output, detector_result)


if __name__ == "__main__":
    print(
        "You may have run the wrong file! \n\t-To run LUMOS as a whole, in terminal type the following commands: \n\t\tcd ../..\n\t\tpython3 main.py\n\n\tTo test this class, please find and run the testing.py in this same directory"
    )
