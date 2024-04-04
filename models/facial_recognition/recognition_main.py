import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import cv2
import numpy as np
import math


class FacialRecognition:
    def generate_response():
        """
        Given a frame image (as ndarray) and text, it will generate a response to any question
        along the lines of:
            - Who is here?
            - Is x here?
            - How many people are here?
            - Is there anybody here I don't know?
        Returns:
            -String resposne to the question
        """
        pass

    def find_rect_of_x():
        """
        Given a string which is the name of the desired person, it will attempt to find the
        person in the given frame/image and return a cropped portion of the image with just this
        individual.

        Returns:
            -An ndarray, which is a cropped portion of the original frame containing only person x's face
            -null =
        """
        pass

    def _classify_faces(self, faces: list) -> list[str]:
        """
        Given a list of cropped faces (as 3D ndarrays), this will classify them if they have already been
        added to the folder /faces/

        If they are not recognised, ??? will be returned
        """

        pass

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
                    int(p["min_y"]) : int(p["max_y"]), int(p["min_x"]) : int(p["max_x"])
                ]
            )

        return (output, detector_result)


if __name__ == "__main__":
    print(
        "You may have run the wrong file! \n\t-To run LUMOS as a whole, in terminal type the following commands: \n\t\tcd ../..\n\t\tpython3 main.py\n\n\tTo test this class, please find and run the testing.py in this same directory"
    )
