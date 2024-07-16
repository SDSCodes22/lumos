import cv2
import numpy as np
from sklearn.cluster import KMeans

def get_dominant_color(image_path, k=3):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape the image to be a list of pixels
    pixels = image.reshape(-1, 3)

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)

    # Get the RGB values of the dominant colors
    dominant_colors = kmeans.cluster_centers_.astype(int)

    return dominant_colors

image_path = "path/to/image.jpg"
dominant_colors = get_dominant_color(image_path, k=3)
print("Dominant Colors:", dominant_colors)