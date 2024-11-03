import requests
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering  # type: ignore
import torch
import numpy as np
import time

device = torch.device("mps")


class BlipTest:
    def __init__(self):
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.model = BlipForQuestionAnswering.from_pretrained(
            "Salesforce/blip-vqa-base"
        )

    def test_blip(self, img: np.ndarray, question: str) -> str:
        # convert to PIL Image
        image = Image.fromarray(np.uint8(img)).convert("RGB")
        # DEBUG
        image.save("test.jpg")
        inputs = self.processor(img, question, return_tensors="pt")  # type: ignore

        out = self.model.generate(  # type: ignore
            **inputs,  # type: ignore
        )  # type: ignore
        print(
            f"Q: {question}, A: {str(self.processor.decode(out[0], skip_special_tokens=True))}"
        )
        return str(self.processor.decode(out[0], skip_special_tokens=True))  # type: ignore


if __name__ == "__main__":
    img_url = (
        "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
    )
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

    question = "Is it safe to walk forwards?"
    print("\n\n\tTiming model!")
    start_time = time.time()
    inputs = processor(raw_image, question, return_tensors="pt")  # type: ignore

    out = model.generate(  # type: ignore
        **inputs,  # type: ignore
    )  # type: ignore
    print(processor.decode(out[0], skip_special_tokens=True))  # type: ignore
    end_time = time.time()

    print(f"\n\nTotal Time Taken: {end_time-start_time} seconds.")
