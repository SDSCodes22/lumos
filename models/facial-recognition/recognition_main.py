class FacialRecognition:
    def generate_response():
        """
        Given a frame image (as ndarray) and text, it will generate a response to any question
        along the lines of:
            - Who is here?
            - Is x here?
            - How many people are here?
            - Is there anybody here I don't know?
        Returns:
            -String resposne to the question
        """
        pass

    def find_rect_of_x():
        """
        Given a string which is the name of the desired person, it will attempt to find the
        person in the given frame/image and return a cropped portion of the image with just this
        individual.

        Returns:
            -An ndarray, which is a cropped portion of the original frame containing only person x's face
            -null =
        """
        pass

    def _find_faces_in_frame():
        """
        Given an image as an ndarray, this will identify all the human faces and return
        a list of all these cropped faces, where each element is a face
        """
        # Initialize library

        # Parse Results into List
        # Crop image for item in list
