
import cv2
import socket
import struct
import numpy as np
import time
import os
import subprocess
from paddleocr import PaddleOCR

# ---------------- CONFIGURATION ----------------

PI_IP = "10.220.139.38"
PORT = 5000

AI_PYTHON = os.path.expanduser("~/smartcap/ai-env/bin/python")
PIPER_MODEL = os.path.expanduser(
    "~/smartcap/voices/en_US-lessac-medium.onnx"
)

AUDIO_FILE = "/tmp/smartcap_ocr_speech.wav"

# ------------------------------------------------


def receive_exact(sock, size):
    data = bytearray()

    while len(data) < size:
        packet = sock.recv(size - len(data))

        if not packet:
            return None

        data.extend(packet)

    return bytes(data)


def speak_text(text):
    """Generate speech using Piper and play it."""

    if not text.strip():
        print("No text available to speak.")
        return

    print("\nGenerating speech...")

    try:
        # Generate speech using Piper
        subprocess.run(
            [
                AI_PYTHON,
                "-m", "piper",
                "-m", PIPER_MODEL,
                "-f", AUDIO_FILE,
                "--", text
            ],
            check=True
        )

        print("Playing recognised text...")

        # Play generated audio through laptop speakers
        subprocess.run(
            ["aplay", "-q", AUDIO_FILE],
            check=True
        )

        print("Speech completed.")

    except subprocess.CalledProcessError as e:
        print("Speech generation/playback failed:", e)

    except FileNotFoundError as e:
        print("Required executable not found:", e)


# ---------------- INITIALIZE OCR ----------------

print("Initializing PaddleOCR...")

ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

print("PaddleOCR initialized successfully!")


# ---------------- CONNECT TO PI ----------------

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

print("\nConnecting to Raspberry Pi camera...")

try:
    client.connect((PI_IP, PORT))
except Exception as e:
    print("Camera connection failed:", e)
    exit()

print("Camera connected successfully!")


# ---------------- MAIN PROGRAM ----------------

print("\n========== SMART CAP OCR + SPEECH ==========")
print("SPACE : Capture, recognise and speak text")
print("Q     : Quit")
print("============================================\n")

try:
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

        cv2.imshow("Smart Cap - OCR and Speech", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == 32:

            print("\nCapturing image...")

            cv2.imwrite("ocr_test.jpg", frame)

            start_time = time.time()

            print("Recognising text...")

            try:
                results = ocr.predict("ocr_test.jpg")

                recognised_lines = []

                print("\n========== OCR RESULTS ==========")

                for result in results:

                    data = result.json

                    if callable(data):
                        data = data()

                    res = data.get("res", data)

                    texts = res.get("rec_texts", [])
                    scores = res.get("rec_scores", [])

                    for text, score in zip(texts, scores):

                        text = str(text).strip()

                        if text:
                            recognised_lines.append(text)

                            print(
                                f"{text} | Confidence: "
                                f"{float(score):.3f}"
                            )

                if recognised_lines:

                    # Combine recognised lines into one speech input
                    full_text = ". ".join(recognised_lines)

                    print("\nComplete recognised text:")
                    print(full_text)

                    print("\nOCR processing time:",
                          round(time.time() - start_time, 2),
                          "seconds")

                    # Speak recognised text
                    speak_text(full_text)

                else:
                    print("No readable text detected.")

                print("\n=================================\n")

            except Exception as e:
                print("OCR error:", e)

finally:
    client.close()
    cv2.destroyAllWindows()
    print("Smart Cap OCR + Speech stopped.")
