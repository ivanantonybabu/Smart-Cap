#!/usr/bin/env python3
"""Local PaddleOCR TCP worker for Smart Cap. Run in ocr-env on the Acer."""
import json
import socket
import struct
import time

import cv2
import numpy as np
from paddleocr import PaddleOCR

HOST = "127.0.0.1"
PORT = 6002
MAX_IMAGE_BYTES = 12 * 1024 * 1024

print("[OCR] Initialising PaddleOCR...")
ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)
print("[OCR] Ready; listening on 127.0.0.1:6002")


def receive_exact(conn, size):
    data = bytearray()
    while len(data) < size:
        part = conn.recv(min(65536, size - len(data)))
        if not part:
            return None
        data.extend(part)
    return bytes(data)


def extract_lines(result):
    lines = []
    all_text = []
    if not result:
        return lines, ""
    for page in result:
        if not isinstance(page, dict):
            continue
        texts = page.get("rec_texts", []) or []
        scores = page.get("rec_scores", []) or []
        for i, text in enumerate(texts):
            score = float(scores[i]) if i < len(scores) else 0.0
            text = str(text).strip()
            if text:
                lines.append({"text": text, "confidence": score})
                all_text.append(text)
    return lines, " ".join(all_text)


server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen(4)
server.settimeout(1.0)

try:
    while True:
        try:
            conn, addr = server.accept()
        except socket.timeout:
            continue
        with conn:
            try:
                conn.settimeout(5.0)
                header = receive_exact(conn, 4)
                if header is None:
                    continue
                size = struct.unpack("!I", header)[0]
                if size <= 0 or size > MAX_IMAGE_BYTES:
                    conn.sendall(struct.pack("!I", 2) + b"{}")
                    continue
                payload = receive_exact(conn, size)
                if payload is None:
                    continue
                image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    response = json.dumps({"text": "", "lines": [], "error": "decode"}).encode()
                else:
                    started = time.time()
                    prediction = ocr.predict(image)
                    lines, combined = extract_lines(prediction)
                    response = json.dumps({
                        "text": combined,
                        "lines": lines,
                        "processing_seconds": round(time.time() - started, 3)
                    }).encode("utf-8")
                conn.sendall(struct.pack("!I", len(response)) + response)
            except (OSError, ValueError, TypeError) as exc:
                print("[OCR] Request error:", exc)
except KeyboardInterrupt:
    print("\n[OCR] Stopping.")
finally:
    server.close()
