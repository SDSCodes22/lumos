import cv2

def person_recognition(image_path):
    # Load the pre-trained Haar cascade for person detection
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

    # Load the image
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect persons in the image
    persons = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected persons
    for (x, y, w, h) in persons:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the image with the detected persons
    cv2.imshow("Person Recognition Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Call the function with the path to your image
person_recognition("path/to/your/image.jpg")