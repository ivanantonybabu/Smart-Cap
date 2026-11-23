
import cv2
import socket
import struct
import numpy as np
import time
from paddleocr import PaddleOCR

# Raspberry Pi camera server
PI_IP = "10.220.139.38"
PORT = 5000

def receive_exact(sock, size):
    data = bytearray()

    while len(data) < size:
        packet = sock.recv(size - len(data))

        if not packet:
            return None

        data.extend(packet)

    return bytes(data)


print("Initializing PaddleOCR...")

ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

print("PaddleOCR initialized!")

# Connect to Raspberry Pi
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

print("Connecting to Raspberry Pi camera...")

try:
    client.connect((PI_IP, PORT))
except Exception as e:
    print("Connection failed:", e)
    exit()

print("Camera connected successfully!")

print("\nPress SPACE to capture and read text.")
print("Press Q to quit.")

while True:

    # Receive frame size
    size_data = receive_exact(client, 4)

    if size_data is None:
        print("Camera disconnected.")
        break

    frame_size = struct.unpack("!I", size_data)[0]

    # Receive JPEG frame
    frame_data = receive_exact(client, frame_size)

    if frame_data is None:
        print("Frame reception failed.")
        break

    # Decode image
    frame = cv2.imdecode(
        np.frombuffer(frame_data, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )

    if frame is None:
        continue

    cv2.imshow("Smart Cap - OCR Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == 32:

        print("\nCapturing image...")

        cv2.imwrite("ocr_test.jpg", frame)

        start_time = time.time()

        print("Running OCR...")

        try:
            results = ocr.predict("ocr_test.jpg")

            print("\n========== OCR RESULTS ==========")

            found_text = False

            for result in results:

                data = result.json

                if callable(data):
                    data = data()

                res = data.get("res", data)

                texts = res.get("rec_texts", [])
                scores = res.get("rec_scores", [])

                for text, score in zip(texts, scores):

                    found_text = True

                    print(
                        f"{text} | Confidence: {float(score):.3f}"
                    )

            if not found_text:
                print("No readable text detected.")

            print(
                "Processing time:",
                round(time.time() - start_time, 2),
                "seconds"
            )

            print("=================================\n")

        except Exception as e:
            print("OCR error:", e)

client.close()
cv2.destroyAllWindows()
