"""
This file uses OpenAI's API to communicate with GPT4-preview-vision.
When using this file, please note that this is all being handled asynchronously
"""

import vertexai
from vertexai.generative_models import GenerativeModel, Image
from pathlib import PurePath
from dotenv import load_dotenv
from os.path import dirname, join

load_dotenv()


#! This class is synchronous
class GPTInteracter:

    def __init__(self):
        # The project is the Google Cloud Project Name
        print("Initializing Vertex AI!")
        vertexai.init(project="lumos-418020", location="europe-west2")
        self.model = GenerativeModel("gemini-1.0-pro-vision")
        print("...\tDone!")

    def get_response(
        self,
        text: str,
    ):
        # Get the image ready
        img_path = join(dirname(dirname(dirname(__file__))), "frame.jpg")

        # Generating a response
        response = self.model.generate_content([text, Image.load_from_file(img_path)])
        print(f"Received response from Gemini.")
        return response.text


# Just a test
if __name__ == "__main__":
    gpt = GPTInteracter()
    response = gpt.get_response("Describe briefly what is in this image.")
    print(response)
