"""
This file uses OpenAI's API to communicate with GPT4-preview-vision.
When using this file, please note that this is all being handled asynchronously
"""

from openai import AsyncOpenAI
from dotenv import load_dotenv
import base64
import asyncio
import aiofiles
import os

load_dotenv()


class GPTInteracter:

    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    async def get_response(self, text: str):
        # TODO: Handle context, I really don't wanna code that at the minute. We'd have to store the messages in a local json file
        response = await self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            #! This is a placeholder that should be tweaked to improve performance
                            "text": "You are an assistant for those who are visually impaired or blind. You will be interacting with users who are visually impaired or blind. The prompts you receive will be from these people. For each prompt, you will be given a question / prompt - most likely relating to the person's surroundings, however this may not always be the case. You will also be given an image, and this image is what this visually impaired person is seeing at the current moment. Given these 2 pieces of data, your task is to generate the best, concise response in which should reference the image in some way, shape or form.",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{self.open_img()}"
                            },
                        },
                    ],
                },
            ],
            max_tokens=300,
        )

        print(f"GPT4 Response: {response.choices[0]}")
        return response.choices[0]

    async def open_img(self, path="test.txt") -> str:
        async with aiofiles.open(path, mode="r") as f:
            contents = await f.read()
        return contents

    async def close_session(self):
        await self.client.close()


# Just a test
if __name__ == "__main__":
    gpt = GPTInteracter()
    resp = asyncio.run(gpt.get_response("What is around me?"))
    asyncio.run(gpt.close_session())
    print(resp)
    print(os.environ.get("OPENAI_API_KEY"))
