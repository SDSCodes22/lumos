from PIL import Image, ImageEnhance, ImageFilter
import pytesseract 

# Define a function to process the image
def process_image(image_path, language='eng'):
    # Open the image
    image = Image.open(image_path)

    # Convert the image to grayscale
    image = image.convert('L')

    # Apply image filters
    image = image.filter(ImageFilter.MedianFilter())

    # Enhance the image contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)

    # Optionally, define a region of interest (ROI)
    # For example, this ROI cuts out the center of the image
    # width, height = image.size
    # roi = image.crop((width/4, height/4, width*3/4, height*3/4))

    # Use image or roi if you want to process a region of interest
    text = pytesseract.image_to_string(image, lang=language)

    return text

# Replace 'path_to_image.jpg' with your image file path
text_output = process_image('path_to_image.jpg', language='eng')

print(text_output)