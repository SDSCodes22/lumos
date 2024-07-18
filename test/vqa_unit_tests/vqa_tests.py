import os
from os.path import join, dirname
import json
import cv2
from typing import Callable
from multiprocessing import Pool, cpu_count
from numpy import ndarray
from sentence_transformers import SentenceTransformer, util  # type: ignore -- REASON: Pylance doesn't detect sentence_transformers
from loguru import logger as log

# -------------------------------------------------------
# |               IMPORTANT NOTICE                        |
# |   These tests will not work without downloading the   |
# |   VQA train2014 dataset. Download this from the       |
# |   VQA v2 website and unzip the file in this directory |
# |   leaving a folder named `train2014`                  |
# |                                                       |
# |   This folder should not be touched.                  |
# |   This code WILL NOT WORK without this being added    |
# |   Licensed under the MIT License, Soumyadeep Saha 2024|
# --------------------------------------------------------


class VQATester:
    def __init__(self) -> None:
        """Initializes class and imports VQA v2 questions from the json file

        Raises:
            ImportError: If the directory `/train2014` is not found.
        """

        # Load the questions
        if not os.path.exists(join(dirname(__file__), "train2014")):
            raise ImportError(
                "Unable to find the train2014/ directory. \nMake sure you have installed the VQA v2 dataset!"
            )

        questions_path = join(dirname(__file__), "questions.json")
        with open(questions_path, "r") as questions:
            temp = json.load(questions)
            self.questions = temp["questions"]
        self.num_questions = len(self.questions)

        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def _test_worker(
        self, model_predict_function: Callable[[ndarray, str], str], question: dict
    ) -> float:
        # Get the image
        img_id = f"{question['image_id']:012}"
        img_path = join(dirname(__file__), "train2014", f"COCO_train2014_{img_id}.jpg")
        try:
            img = cv2.imread(img_path)
        except:
            log.warning(
                f"Image doesn't exist.\t ID: {img_id}, Path searched: {img_path}. Worker cancelling"
            )
            return 0.0

        # Feed into the model
        prediction: str = model_predict_function(img, question["question"])

        # Compute similarity scores
        embedding_pred = self.model.encode(prediction, convert_to_tensor=True)
        embedding_true = self.model.encode(question["question"], convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embedding_pred, embedding_true)

        return similarity.item()

    def test(
        self,
        model_predict_function: Callable[[ndarray, str], str],
        percent_to_use: int = 50,
    ):
        pass
