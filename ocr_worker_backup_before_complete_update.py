
import socket
import struct
import json
import cv2
import numpy as np
import os
import time

from paddleocr import PaddleOCR

HOST = "127.0.0.1"
PORT = 6002

IMAGE_FILE = "/tmp/smartcap_ocr_input.jpg"
MAX_IMAGE_SIZE = 10 * 1024 * 1024

def receive_exact(conn, size):
    data = bytearray()

    while len(data) < size:
        packet = conn.recv(min(4096, size - len(data)))

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

print("PaddleOCR initialized successfully.")
print("OCR device:", __import__("paddle").device.get_device())

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen(5)

print(f"OCR worker listening on {HOST}:{PORT}")

try:
    while True:
        conn, address = server.accept()

        with conn:
            try:
                header = receive_exact(conn, 4)

                if header is None:
                    continue

                image_size = struct.unpack("!I", header)[0]

                if image_size <= 0 or image_size > MAX_IMAGE_SIZE:
                    raise ValueError("Invalid image size")

                image_data = receive_exact(conn, image_size)

                if image_data is None:
                    continue

                with open(IMAGE_FILE, "wb") as f:
                    f.write(image_data)

                start_time = time.time()

                results = ocr.predict(IMAGE_FILE)

                recognised_lines = []

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
                            recognised_lines.append({
                                "text": text,
                                "confidence": float(score)
                            })

                full_text = ". ".join(
                    item["text"] for item in recognised_lines
                )

                response = {
                    "text": full_text,
                    "lines": recognised_lines,
                    "processing_time": round(
                        time.time() - start_time, 2
                    )
                }

                response_data = json.dumps(response).encode("utf-8")

                conn.sendall(struct.pack("!I", len(response_data)))
                conn.sendall(response_data)

                print("\nOCR completed:", full_text)
                print("Processing time:", response["processing_time"], "seconds")

            except Exception as error:
                print("OCR worker error:", error)

                response_data = json.dumps({
                    "text": "",
                    "lines": [],
                    "error": str(error)
                }).encode("utf-8")

                try:
                    conn.sendall(struct.pack("!I", len(response_data)))
                    conn.sendall(response_data)
                except OSError:
                    pass

except KeyboardInterrupt:
    print("\nStopping OCR worker.")

finally:
    server.close()
    if os.path.exists(IMAGE_FILE):
        os.remove(IMAGE_FILE)
