import os
from os.path import join, dirname
import json
from xml.dom.expatbuilder import theDOMImplementation
import cv2

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
