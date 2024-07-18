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
import os
from os.path import join, dirname
import json
import cv2
from typing import Callable
from numpy import ndarray
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util  # type: ignore
from loguru import logger as log
import sys


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
        log.debug("VQATester class initialization complete.")

    def _test_worker(
        self,
        model_predict_function: Callable[[ndarray, str], str],
        question: dict,
    ) -> float:
        """Given the model to test, and the questions list,
        computes similarity score using cosine similarity between the prediction and the true values.

        Args:
            model_predict_function (Callable[[ndarray, str], str): A function which uses your model to complete the VQA Task.
                Must expect an ndarray (the image) and a str (the question). Must return a string, which is the predicted answer
            questions (list): List of questions to process
            results_queue (mp.Queue): Queue to store the results

        """

        img_id = f"{question['image_id']:012}"
        img_path = join(dirname(__file__), "train2014", f"COCO_train2014_{img_id}.jpg")

        img = cv2.imread(img_path)
        if img is None:
            log.warning(
                f"Image doesn't exist.\t ID: {img_id}, Path searched: {img_path}. Worker skipping"
            )
            return 0.0

        # Feed into the model
        prediction: str = model_predict_function(img, question["question"])

        # Compute similarity scores
        embedding_pred = self.model.encode(prediction, convert_to_tensor=True)
        embedding_true = self.model.encode(question["question"], convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embedding_pred, embedding_true)
        return similarity.item()

    def center_text(self, text):
        # Get the current terminal width
        terminal_width = os.get_terminal_size().columns

        # Split the text into lines
        lines = text.split("\n")

        # Center each line and join them back together
        centered_lines = [line.center(terminal_width) for line in lines]
        return "\n".join(centered_lines)

    def test(
        self,
        model_predict_function: Callable[[ndarray, str], str],
        percent_to_use: int = 10,
        num_processes=None,
    ) -> float:
        """Function to test a model's accuracy on the VQA v2 Dataset. Provides interactive usage

        Args:
            model_predict_function (Callable[[ndarray (the image), str (the question)], str (the prediction)]): The "predict" function of your model. Used to test your model
            percent_to_use (int, optional): How much of the total training data of VQA v2 do you want to test on? Write 100 to test on the full dataset. Defaults to 10.
            num_processes (_type_, optional): Number of processes to use for the pool. Defaults to number of cpus.

        Returns:
            float: The mean score, if you need to use it. Also prints out the results.
        """
        num_questions = int((percent_to_use / 100) * self.num_questions)

        results = []
        # Use TQDM for an appealing, and informative progress bar
        for i in tqdm(range(num_questions), desc="Test Progress"):
            x = self._test_worker(model_predict_function, self.questions[i])
            results.append(x)

        print(
            self.center_text(
                f"\n\n\nDone!\nMean Similarity Score: {sum(results) / len(results):.5f}\nAs Percentage: {sum(results) / len(results) * 100:.2f}%\n\n\n"
            )
        )

        return sum(results) / len(results)


#           HERE WE DO EXAMPLE USAGE ON 1% OF THE VQA DATASET
if __name__ == "__main__":
    # Define a dummy model predict function for demonstration
    def dummy_model_predict_function(img: ndarray, question: str) -> str:
        # Dummy prediction logic
        return "(Gibberish) (Gibberish) Oopsie Daises!"

    # ? Example Usage
    tester = VQATester()
    tester.test(dummy_model_predict_function, 1)
