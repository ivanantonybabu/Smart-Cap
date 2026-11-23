
import pytesseract
from PIL import Image

# Test OCR using an image file
image = Image.open("test_text.jpg")

text = pytesseract.image_to_string(
    image,
    lang="eng",
    config="--psm 6"
)

print("\nRecognized text:")
print("----------------")
print(text)
