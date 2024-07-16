import easyocr
import os
from os import path
import numpy.typing as npt
from loguru import logger as log
import numpy as np
import cv2
from typing import Union

# from ultralytics import YOLO
import vertexai
from vertexai.generative_models import GenerativeModel, Image, Part, Content
from pathlib import PurePath
from dotenv import load_dotenv


load_dotenv()


class SceneTextRecognition:
    def __init__(self):
        self.reader = easyocr.Reader(["en"])
        # self.yolo = YOLO("yolov8m.pt")
        # The project is the Google Cloud Project Name
        print("Initializing Vertex AI!")
        vertexai.init(project="lumos-418020", location="europe-west2")
        self.model = GenerativeModel("gemini-1.0-pro-vision")
        print("...\tDone!")

    def read_text(self, frame) -> dict:
        log.info("reading image")
        result = self.reader.readtext(frame)
        if len(result) == 0:
            return {"No text was detected": [0, 0]}, "No text was detected"
        log.info("Done.")
        # We need to sort the results as they mightn't be in the best order
        # We need to format the array so that it can be sorted
        # Use a dict!
        wordBoxKey = dict()
        totalWordHeight = 0
        boxes = []  # Top left coordinate of boxes eg.[[4, 5], [5, 6], ...]
        for i, x in enumerate(result):
            log.info(f"Example of result: {x}")
            # Store key as x coordinate of top left of bounding box * y coord
            wordBoxKey[x[0][0][0] * x[0][0][1]] = x[1]
            totalWordHeight += x[0][0][1] - x[0][2][1]
            boxes.append(x[0][0])
        totalWordHeight /= len(result)
        log.info(f"Created hashmap: {wordBoxKey}\n\n Also created boxes: {boxes}")
        # Sort by Y using numpy
        boxes = np.array(boxes)
        inds = np.argsort(boxes[:, 1])
        boxes = boxes[inds]

        log.info(f"sorted by Y: {boxes}")
        # Where the change from in to in+1 is more than x times bigger than the average change previously
        # , Split the list to create rows
        PERCENTAGE_OVER_MARGIN = 0.5  # What % over the average accounts for a new line
        rows = []
        row = []  # The current row being parsed
        total_y = 0  # Sum of all Y values parsed in the row
        count = 0  # Number of point in the row so far
        last_y = 0  # The last Y value
        avg_y = 0  # The average Y value
        for i in boxes:
            if abs(last_y - i[1]) >= PERCENTAGE_OVER_MARGIN * totalWordHeight + avg_y:
                # Not on the same row, We need to split this row
                rows.append(row)
                row = [i]
                count = 1
                avg_y = i[1]
                total_y = i[1]
                last_y = i[1]
            else:
                # update values
                last_y = i[1]
                total_y += i[1]
                avg_y = total_y / count
                count += 1

                # add value to current row
                row.append(i)
        # append remaining rows to row
        rows.append(row)
        # index 0 will be empty so pop
        rows.pop(0)
        log.info(f"Split into rows: {rows}")
        # In each row, sort by the X
        for i, row in enumerate(rows):
            arr = np.array(row)
            inds = np.argsort(arr[:, 0])
            rows[i] = arr[inds].tolist()
        log.info(f"Rows sorted by x: {rows[0]}")
        # Parse text
        # using hashmap from before
        words = dict()
        sentence = ""
        for y, row in enumerate(rows):
            for x, box in enumerate(row):
                word = f"{wordBoxKey[box[0] * box[1]]} "
                sentence += word
                words[word] = box
            sentence += "\n"

        print("RESULT: ", words)
        # example entry: "{"BLACK": [x_coord, y_coord]}"
        return words, sentence

    def find_objects(self, frame) -> list:
        results = self.yolo.predict(frame)
        return results

    def is_in_rectangle(point: list, top_left: list, bottom_right: list) -> bool:
        good_x = top_left[0] <= point[0] and point[0] <= bottom_right[0]
        good_y = top_left[1] <= point[1] and point[1] <= bottom_right[1]
        return good_x and good_y

    def create_json(self, objects, text) -> str:
        output = []
        # Loop through all points on x
        for word in text:
            for result in objects:
                box = result.boxes[0].xyxy.to_list()
                if self.is_in_rectangle(text[word], box[:2], box[2:]):
                    # word is in a detected object
                    object_name = result.names[result.boxes[0].cls[0].item()]
                    response = {"word": word, "object": object_name}
                    output.append(response)
        return str(output)

    def generate_response(self, frame, text: str = "No prompt was given") -> str:
        text_boxes = self.read_text(frame)
        # object_boxes = self.find_objects(frame)

        # raw_json = self.create_json(object_boxes, text_boxes)

        text_generated = f"""
        You are an AI Assistant who's role is to assist blind people by analyzing their surroundings. The image attached is from the POV of the blind person. A blind person has asked you the following prompt: {text}. \n 
        To assist you to answer this prompt, an OCR model has interpreted and extracted the text for you and here it is :
        {text_boxes[1]} \n 
        Now, use this to assist you answer the original prompt. Do not use language like 'in the picture'. Replace this with language like 'in front of you'. Do not directly reference the json or the OCR. It is only there to assist
        you with your propt generation. Do you understand? If so, concisely reply to the original prompt: {text}
        """

        return self.get_gemini_response(text_generated, frame)

    def get_gemini_response(self, text: str, img: npt.NDArray):
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


if __name__ == "__main__":
    sceneTextRecognition = SceneTextRecognition()
    img = os.path.join(os.path.dirname(__file__), "test_2.webp")
    print(f"RESULT (SORTED): {sceneTextRecognition.generate_response(img)}")

    import cv2

    image = cv2.imread("test_2.webp", 0)
    thresh = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2
    )

    cv2.imwrite("temp.png", thresh)
    img = os.path.join(os.path.dirname(__file__), "temp.png")
    print(f"RESULT (SORTED): {sceneTextRecognition.generate_response(img)}")
    cv2.imshow("thresh", thresh)
    cv2.waitKey(0)
