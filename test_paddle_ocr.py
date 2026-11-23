
import cv2
from paddleocr import PaddleOCR
import time

print("Initializing PaddleOCR...")

ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

print("PaddleOCR initialized!")

# Open camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Laptop camera not available.")
    print("We will use an image instead.")
    exit()

print("\nPress SPACE to capture an image.")
print("Press Q to quit.")

while True:
    ret, frame = camera.read()

    if not ret:
        print("Camera frame unavailable")
        break

    cv2.imshow("PaddleOCR Test - Press SPACE", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == 32:
        print("\nCapturing image...")
        start = time.time()

        cv2.imwrite("ocr_test.jpg", frame)

        results = ocr.predict("ocr_test.jpg")

        print("\n========== OCR RESULTS ==========")

        for result in results:
            data = result.json
            if callable(data):
                data = data()

            res = data.get("res", data)

            texts = res.get("rec_texts", [])
            scores = res.get("rec_scores", [])

            for text, score in zip(texts, scores):
                print(f"{text}  | Confidence: {score:.3f}")

        print("Processing time:", round(time.time() - start, 2), "seconds")
        print("=================================\n")

camera.release()
cv2.destroyAllWindows()
