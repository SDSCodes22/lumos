"""
To debug or test issues with the Facial Recognition class, run this file.
"""

from recognition_main import FacialRecognition_Lumos
import os
import cv2
import numpy as np
import math

import vertexai
from vertexai.generative_models import GenerativeModel, Image, Part, Content
from pathlib import PurePath
from dotenv import load_dotenv
import cv2
from os.path import dirname, join
import numpy.typing as npt

load_dotenv()


#! This class is synchronous
class GPTInteracter:

    def __init__(self):
        # The project is the Google Cloud Project Name
        print("Initializing Vertex AI!")
        vertexai.init(project="lumos-418020", location="europe-west2")
        self.model = GenerativeModel("gemini-1.0-pro-vision")
        print("...\tDone!")

    def get_response(self, text: str, img: npt.NDArray):
        # Get the image ready - we need to encode as jpg, then get bytes, then convert to vision.Image
        _, encoded = cv2.imencode(".jpg", img)
        img = encoded.tobytes()
        # Generating a response
        response = self.model.generate_content(
            contents=[
                Content(
                    role="user",
                    parts=[
                        Part.from_text(text),
                        Part.from_image(Image.from_bytes(img)),
                    ],
                )
            ]
        )
        print(f"Received response from Gemini.")
        return response.text


# Just a test
if __name__ == "__main__":
    gpt = GPTInteracter()
    path = join(dirname(dirname(dirname(__file__))), "frame.jpg")
    response = gpt.get_response(
        "Describe briefly what is in this image.", cv2.imread(path)
    )
    print(response)
# # Copied from Google Docs for Testing
# MARGIN = 10  # pixels
# ROW_SIZE = 10  # pixels
# FONT_SIZE = 1
# FONT_THICKNESS = 1
# TEXT_COLOR = (255, 0, 0)  # red


# def _normalized_to_pixel_coordinates(
#     normalized_x: float, normalized_y: float, image_width: int, image_height: int
# ):
#     """Converts normalized value pair to pixel coordinates."""

#     # Checks if the float value is between 0 and 1.
#     def is_valid_normalized_value(value: float) -> bool:
#         return (value > 0 or math.isclose(0, value)) and (
#             value < 1 or math.isclose(1, value)
#         )

#     if not (
#         is_valid_normalized_value(normalized_x)
#         and is_valid_normalized_value(normalized_y)
#     ):
#         # TODO: Draw coordinates even if it's outside of the image bounds.
#         return None
#     x_px = min(math.floor(normalized_x * image_width), image_width - 1)
#     y_px = min(math.floor(normalized_y * image_height), image_height - 1)
#     return x_px, y_px


# def visualize(image, detection_result) -> np.ndarray:
#     """Draws bounding boxes and keypoints on the input image and return it.
#     Args:
#         image: The input RGB image.
#         detection_result: The list of all "Detection" entities to be visualize.
#     Returns:
#         Image with bounding boxes.
#     """
#     annotated_image = image.copy()
#     height, width, _ = image.shape

#     for detection in detection_result.detections:
#         # Draw bounding_box
#         bbox = detection.bounding_box
#         start_point = bbox.origin_x, bbox.origin_y
#         end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
#         cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

#         # Draw keypoints
#         for keypoint in detection.keypoints:
#             keypoint_px = _normalized_to_pixel_coordinates(
#                 keypoint.x, keypoint.y, width, height
#             )
#             color, thickness, radius = (0, 255, 0), 2, 2
#             cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

#         # Draw label and score
#         category = detection.categories[0]
#         category_name = category.category_name
#         category_name = "" if category_name is None else category_name
#         probability = round(category.score, 2)
#         result_text = category_name + " (" + str(probability) + ")"
#         text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
#         cv2.putText(
#             annotated_image,
#             result_text,
#             text_location,
#             cv2.FONT_HERSHEY_PLAIN,
#             FONT_SIZE,
#             TEXT_COLOR,
#             FONT_THICKNESS,
#         )

#     return annotated_image


recogniser = FacialRecognition_Lumos(GPTInteracter())
img_path = os.path.join(os.path.dirname(__file__), "test_image.jpeg")

# print("\t TEST 1: Find Faces in Image")
# print("d: sending...")

img = cv2.imread(img_path)
# images = recogniser._find_faces_in_frame(img)
# # Save the first guy
# if not os.path.exists("faces/test_guy.jpg"):
#     cv2.imwrite("faces/test_guy.jpg", images[0][0])
# print("d: done")
# image = mp.Image.create_from_file(img_path)
# image_copy = np.copy(image.numpy_view())
# annotated_image = visualize(image_copy, images[1])
# rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
# cv2.imshow("title", rgb_annotated_image)
# cv2.waitKey()

print("\t TEST 2: Test main function")
resp = recogniser.generate_response("Anybody here you do not recognise?", img)
print(resp)
