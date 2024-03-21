"""
This file uses OpenAI's API to communicate with GPT4-preview-vision.
When using this file, please note that this is all being handled asynchronously
"""

from openai import AsyncOpenAI
from dotenv import load_dotenv
import base64
import asyncio
import os

load_dotenv()


class GPTInteracter:

    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    async def get_response(self, text: str):

        pass

    async def close_session(self):
        await self.session.close()


# Just a test
if __name__ == "__main__":
    gpt = GPTInteracter()
    img = asyncio.run(
        gpt._encode_image(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "frame.jpg",
            )
        )
    )
    save_string_to_file(img, "test.txt")
    print(os.environ.get("OPENAI_API_KEY"))
