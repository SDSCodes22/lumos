# -------------------------------------------------------
# |               IMPORTANT NOTICE                        |
# |   These tests will not work without downloading the   |
# |   VQA train2014 dataset. Download this from the       |
# |   VQA v2 website and unzip the file in this directory |
# |   leaving a folder named `train2014`. You must also   |
# |   download the train2017 annotations.
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
from tqdm import tqdm  # type: ignore I DO EXIST
from sentence_transformers import SentenceTransformer, util  # type: ignore
from loguru import logger as log  # type: ignore pylance no like modules fr
import sys
from collections import defaultdict

# IMPORT YOUR MODEL HERE
from blip import BlipTest  # type: ignore

log.debug("Finished all imports!")


class VQATester:
    def __init__(self) -> None:
        """Initializes class and imports VQA v2 questions from the json file

        Raises:
            ImportError: If the directory `/train2014` is not found.
        """
        self.question_types = defaultdict(list)
        self.answer_types = {"yes/no": [], "number": [], "other": []}
        # Load the questions
        if not os.path.exists(join(dirname(__file__), "train2014")):
            raise ImportError(
                "Unable to find the train2014/ directory. \nMake sure you have installed the VQA v2 dataset!"
            )

        questions_path = join(dirname(__file__), "questions.json")
        answers_path: str = join(dirname(__file__), "answers.json")
        log.debug("Loading Questions...")
        with open(questions_path, "r") as questions:
            temp = json.load(questions)
            self.questions = temp["questions"]
        self.num_questions = len(self.questions)
        log.debug("Loading Answers...")
        with open(answers_path, "r") as answers:
            temp = json.load(answers)
            self.answers = temp["annotations"]
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
        # Get the true answer
        question_id = question["question_id"]
        true_ans, question_type, answer_type = self.get_answer(int(question_id))  # type: ignore Trust

        # Compute similarity scores
        embedding_pred = self.model.encode(prediction, convert_to_tensor=True)
        embedding_true = self.model.encode(true_ans, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embedding_pred, embedding_true)

        self.answer_types[answer_type].append(similarity.item())
        self.question_types[question_type].append(similarity.item())

        print(f"Q: {question['question']}, True: {true_ans}, Pred: {prediction}")
        return similarity.item()

    def get_answer(self, question_id: int):
        """Use binary search to search for a particular answer in the VQA v2 dataset, given the question ID

        Args:
            question_id (int): Question ID

        Returns:
            tuple : (Answer to question [str], Question Type [str], Answer Type [str])
        """
        # Return the answer, the question type, and the answer type
        # Do a binary search
        left = 0
        right = len(self.answers) - 1
        while left <= right:
            if self.answers[left]["question_id"] == question_id:
                return (
                    self.answers[left]["answers"][0]["answer"],
                    self.answers[left]["question_type"],
                    self.answers[left]["answer_type"],
                )
            elif self.answers[right]["question_id"] == question_id:
                return (
                    self.answers[right]["answers"][0]["answer"],
                    self.answers[right]["question_type"],
                    self.answers[right]["answer_type"],
                )
            left += 1
            right -= 1

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
        percent_to_use: float = 10,
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
        loop_obj = tqdm(range(num_questions), desc="Test Progress")
        # Use TQDM for an appealing, and informative progress bar
        for i in loop_obj:
            x = self._test_worker(model_predict_function, self.questions[i])
            results.append(x)
            if i % 10 == 0 and i != 0:
                print(
                    self.center_text(
                        f"Current mean similarity score: {sum(results) / len(results)}"
                    )
                )

        print(
            self.center_text(
                f"\n\n\nDone!\nMean Similarity Score: {sum(results) / len(results):.5f}\nAs Percentage: {sum(results) / len(results) * 100:.2f}%\n\n\n"
            )
        )
        print(
            self.center_text(
                "Question Types:\n"
                + "\n".join(
                    [f"{k}: {sum(v)/len(v)}" for k, v in self.question_types.items()]
                )
            )
        )
        print(
            self.center_text(
                "Answer Types:\n"
                + "\n".join(
                    [f"{k}: {sum(v)/len(v)}" for k, v in self.answer_types.items()]
                )
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
    test_class = BlipTest()
    tester.test(test_class.test_blip, 0.1)
