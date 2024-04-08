"""
This file uses OpenAI's API to communicate with GPT4-preview-vision.
When using this file, please note that this is all being handled asynchronously
"""

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
