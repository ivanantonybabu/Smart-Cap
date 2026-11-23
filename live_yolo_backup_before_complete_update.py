import cv2
import socket
import struct
import numpy as np
import queue
import time
import threading
import subprocess
import torch
import sys
import os
import json

from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================

PI_IP = "10.220.139.38"

CAMERA_PORT = 5000
AUDIO_PORT = 6000
MODE_PORT = 6001
OCR_PORT = 6002

MODEL_NAME = "yolo11s.pt"
IMAGE_SIZE = 640
CONFIDENCE = 0.30
ANNOUNCEMENT_COOLDOWN = 5.0

# OCR is requested periodically while OCR mode is active.
OCR_INTERVAL = 3.0
OCR_REPEAT_COOLDOWN = 20.0
MAX_OCR_IMAGE_SIZE = 10 * 1024 * 1024

PIPER_MODEL = os.path.expanduser(
    "~/smartcap/voices/en_US-lessac-medium.onnx"
)
PIPER_OUTPUT = "/tmp/smartcap_speech.wav"

# ============================================================
# GPU AND YOLO INITIALIZATION
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is not available in ai-env!")

torch.backends.cudnn.benchmark = True
torch.set_num_threads(4)

print("YOLO GPU:", torch.cuda.get_device_name(0))
print("\nLoading YOLO11s...")

model = YOLO(MODEL_NAME)
print("YOLO model loaded successfully.")

print("Warming up YOLO...")
model.predict(
    np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
    device=0,
    imgsz=IMAGE_SIZE,
    half=True,
    verbose=False
)
print("YOLO GPU ready.")

# ============================================================
# SHARED STATE
# ============================================================

latest_frame = None
frame_number = 0
frame_lock = threading.Lock()
frame_ready = threading.Event()
stream_stopped = threading.Event()

stream_fps = 0.0
inference_fps = 0.0

current_mode = "OBJECT"
mode_lock = threading.Lock()

ocr_queue = queue.Queue(maxsize=1)
speech_queue = queue.Queue(maxsize=3)

last_announced = {}
last_ocr_text = ""
last_ocr_time = 0.0

# ============================================================
# MODE RECEIVER: RASPBERRY PI -> ACER, TCP 6001
# ============================================================

def get_mode():
    with mode_lock:
        return current_mode


def mode_receiver():
    global current_mode

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        server.bind(("0.0.0.0", MODE_PORT))
        server.listen(5)
        server.settimeout(1.0)

        print(f"Mode receiver listening on port {MODE_PORT}.")

        while not stream_stopped.is_set():
            try:
                conn, address = server.accept()
            except socket.timeout:
                continue

            with conn:
                try:
                    conn.settimeout(3)
                    data = conn.recv(1024).decode("utf-8").strip().upper()

                    if data in ("OBJECT", "OCR"):
                        with mode_lock:
                            changed = (current_mode != data)
                            current_mode = data

                        if changed:
                            if data == "OBJECT":
                                print("\n========== OBJECT DETECTION MODE ==========")
                            else:
                                print("\n============== OCR READING MODE ==============")

                        conn.sendall(b"OK")
                    else:
                        conn.sendall(b"ER")

                except Exception as error:
                    print("Mode command error:", error)

    except OSError as error:
        if not stream_stopped.is_set():
            print("Mode receiver could not start:", error)
            print("Check whether another program is using port 6001.")
    finally:
        server.close()


# ============================================================
# PIPER SPEECH AND AUDIO TRANSFER
# ============================================================

def send_audio_to_pi(audio_path):
    try:
        with open(audio_path, "rb") as f:
            audio_data = f.read()

        if not audio_data:
            print("Audio file is empty.")
            return False

        with socket.create_connection((PI_IP, AUDIO_PORT), timeout=10) as client:
            client.settimeout(120)
            client.sendall(struct.pack("!I", len(audio_data)))
            client.sendall(audio_data)

            response = bytearray()
            while len(response) < 2:
                packet = client.recv(2 - len(response))
                if not packet:
                    break
                response.extend(packet)

            if bytes(response) == b"OK":
                print("Audio played through Raspberry Pi headphones.")
                return True

            print("Pi audio receiver returned an error.")
            return False

    except Exception as error:
        print("Audio transfer error:", error)
        return False


def speech_worker():
    print("Piper speech worker started.")

    while True:
        message = speech_queue.get()

        try:
            if message is None:
                return

            result = subprocess.run(
                [
                    sys.executable, "-m", "piper",
                    "-m", PIPER_MODEL,
                    "-f", PIPER_OUTPUT,
                    "--", message
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True
            )

            if result.returncode != 0:
                print("Piper generation error:")
                print(result.stderr)
                continue

            print("Sending speech to Raspberry Pi...")
            send_audio_to_pi(PIPER_OUTPUT)

        except Exception as error:
            print("Speech system error:", error)
        finally:
            speech_queue.task_done()


speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

# ============================================================
# COMMON NETWORK HELPER
# ============================================================

def receive_exact(sock, size):
    data = bytearray()

    while len(data) < size:
        packet = sock.recv(min(4096, size - len(data)))
        if not packet:
            return None
        data.extend(packet)

    return bytes(data)


# ============================================================
# OBJECT COLOUR DETECTION
# ============================================================

def get_object_color(frame, coordinates):
    x1, y1, x2, y2 = map(int, coordinates)
    height, width = frame.shape[:2]

    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        return "unknown"

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return "unknown"

    h, w = roi.shape[:2]
    if h > 10 and w > 10:
        margin_x = int(w * 0.15)
        margin_y = int(h * 0.15)
        roi = roi[margin_y:h-margin_y, margin_x:w-margin_x]

    if roi.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)

    saturation = pixels[:, 1]
    value = pixels[:, 2]

    valid = pixels[(saturation > 45) & (value > 45)]

    if len(valid) < 20:
        median_value = np.median(value)
        median_sat = np.median(saturation)

        if median_value < 60:
            return "black"
        elif median_value > 190 and median_sat < 45:
            return "white"
        elif median_sat < 45:
            return "grey"
        return "unknown"

    hues = valid[:, 0]
    histogram = np.bincount(hues.astype(np.uint8), minlength=180)
    dominant_hue = int(np.argmax(histogram))

    if dominant_hue <= 10 or dominant_hue >= 170:
        return "red"
    elif dominant_hue <= 22:
        return "orange"
    elif dominant_hue <= 35:
        return "yellow"
    elif dominant_hue <= 85:
        return "green"
    elif dominant_hue <= 100:
        return "cyan"
    elif dominant_hue <= 135:
        return "blue"
    elif dominant_hue <= 160:
        return "purple"
    return "pink"


# ============================================================
# OBJECT SPEECH ANNOUNCEMENTS
# ============================================================

def announce_objects(objects):
    now = time.time()
    to_announce = []

    for obj, color in objects:
        key = (obj, color)
        previous = last_announced.get(key, 0)

        if now - previous >= ANNOUNCEMENT_COOLDOWN:
            to_announce.append(f"{color} {obj}")
            last_announced[key] = now

    if to_announce:
        message = "I can see " + ", ".join(to_announce)
        print("Speaking:", message)

        try:
            speech_queue.put_nowait(message)
        except queue.Full:
            print("Speech queue full. Object announcement skipped.")


# ============================================================
# OCR REQUEST: MAIN AI PROCESS -> OCR WORKER, TCP 6002
# ============================================================

def request_ocr(frame):
    try:
        ok, encoded = cv2.imencode(
            ".jpg", frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        )

        if not ok:
            print("Could not encode frame for OCR.")
            return None

        image_data = encoded.tobytes()

        if len(image_data) > MAX_OCR_IMAGE_SIZE:
            print("OCR image exceeds size limit.")
            return None

        with socket.create_connection(("127.0.0.1", OCR_PORT), timeout=5) as client:
            client.settimeout(90)
            client.sendall(struct.pack("!I", len(image_data)))
            client.sendall(image_data)

            header = receive_exact(client, 4)
            if header is None:
                print("OCR worker closed connection without a response.")
                return None

            response_size = struct.unpack("!I", header)[0]
            if response_size <= 0 or response_size > 2 * 1024 * 1024:
                print("Invalid OCR response size.")
                return None

            response_data = receive_exact(client, response_size)
            if response_data is None:
                print("Incomplete OCR response.")
                return None

            return json.loads(response_data.decode("utf-8"))

    except Exception as error:
        print("OCR communication error:", error)
        return None


def ocr_worker_client():
    global last_ocr_text, last_ocr_time

    print("OCR client thread started.")

    while not stream_stopped.is_set():
        try:
            frame = ocr_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        try:
            if get_mode() != "OCR":
                continue

            response = request_ocr(frame)

            if not response:
                continue

            if response.get("error"):
                print("OCR worker error:", response["error"])
                continue

            text = str(response.get("text", "")).strip()

            if not text:
                print("OCR: No readable text detected.")
                continue

            print("\n========== OCR RESULT ==========")
            print(text)
            print("OCR processing time:", response.get("processing_time", "?"), "seconds")
            print("================================")

            now = time.time()

            # Avoid repeatedly speaking the same visible text.
            if text == last_ocr_text and now - last_ocr_time < OCR_REPEAT_COOLDOWN:
                continue

            # Do not speak a result if the user has switched back to object mode.
            if get_mode() != "OCR":
                continue

            last_ocr_text = text
            last_ocr_time = now

            try:
                speech_queue.put_nowait(text)
            except queue.Full:
                print("Speech queue full. OCR speech skipped.")

        finally:
            ocr_queue.task_done()


# ============================================================
# BACKGROUND CAMERA RECEIVER
# ============================================================

def camera_receiver(sock):
    global latest_frame, frame_number, stream_fps

    received_count = 0
    fps_start = time.time()

    print("Camera receiver thread started.")

    try:
        while not stream_stopped.is_set():
            header = receive_exact(sock, 4)
            if header is None:
                break

            frame_size = struct.unpack("!I", header)[0]

            if frame_size <= 0 or frame_size > 5_000_000:
                print("Invalid camera frame size.")
                break

            frame_data = receive_exact(sock, frame_size)
            if frame_data is None:
                break

            frame = cv2.imdecode(
                np.frombuffer(frame_data, dtype=np.uint8),
                cv2.IMREAD_COLOR
            )

            if frame is None:
                continue

            received_count += 1
            now = time.time()

            if now - fps_start >= 1.0:
                stream_fps = received_count / (now - fps_start)
                received_count = 0
                fps_start = now

            with frame_lock:
                latest_frame = frame
                frame_number += 1
                frame_ready.set()

    except (ConnectionError, OSError) as error:
        if not stream_stopped.is_set():
            print("Camera connection error:", error)

    finally:
        stream_stopped.set()
        frame_ready.set()
        print("Camera receiver stopped.")


# ============================================================
# CONNECT TO RASPBERRY PI CAMERA
# ============================================================

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

print("\nConnecting to Raspberry Pi camera...")

try:
    client.connect((PI_IP, CAMERA_PORT))
except OSError as error:
    print("Could not connect to Raspberry Pi camera:", error)
    stream_stopped.set()
    try:
        speech_queue.put(None, timeout=2)
    except queue.Full:
        pass
    raise SystemExit(1)

print("Camera connected successfully.")

# ============================================================
# START BACKGROUND THREADS
# ============================================================

receiver_thread = threading.Thread(
    target=camera_receiver,
    args=(client,),
    daemon=True
)
receiver_thread.start()

mode_thread = threading.Thread(target=mode_receiver, daemon=True)
mode_thread.start()

ocr_thread = threading.Thread(target=ocr_worker_client, daemon=True)
ocr_thread.start()

print("\n==============================================")
print("SMART CAP - YOLO + PADDLEOCR + PIPER")
print("Initial mode: OBJECT DETECTION")
print("Press Play/Pause on Soundcore Q20i to switch.")
print("OBJECT mode: YOLO + colour identification")
print("OCR mode: PaddleOCR text reading")
print("Q: Quit")
print("==============================================\n")

# ============================================================
# MAIN PROCESSING LOOP
# ============================================================

previous_frame_number = -1
fps_counter = 0
fps_start = time.time()
inference_fps = 0.0
last_ocr_submit = 0.0

try:
    while not stream_stopped.is_set():
        frame_ready.wait(timeout=1.0)

        if stream_stopped.is_set():
            break

        with frame_lock:
            if latest_frame is None:
                frame_ready.clear()
                continue

            if frame_number == previous_frame_number:
                frame_ready.clear()
                continue

            frame = latest_frame.copy()
            current_frame_number = frame_number
            frame_ready.clear()

        previous_frame_number = current_frame_number
        mode = get_mode()

        if mode == "OBJECT":
            annotated_frame = frame.copy()

            results = model.track(
                source=frame,
                device=0,
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE,
                tracker="bytetrack.yaml",
                persist=True,
                half=True,
                max_det=30,
                verbose=False
            )

            result = results[0]
            detected_objects = set()

            if result.boxes is not None:
                for box in result.boxes:
                    coordinates = box.xyxy[0].cpu().numpy()
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    if confidence < CONFIDENCE:
                        continue

                    object_name = model.names[class_id]
                    color = get_object_color(frame, coordinates)
                    detected_objects.add((object_name, color))

                    x1, y1, x2, y2 = map(int, coordinates)
                    label = f"{color} {object_name} {confidence:.2f}"

                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2
                    )

            # Avoid speaking an object result if mode changed during inference.
            if get_mode() == "OBJECT":
                announce_objects(detected_objects)

            display_mode = "OBJECT DETECTION"

        else:
            annotated_frame = frame.copy()
            display_mode = "OCR READING"

            now = time.time()

            if now - last_ocr_submit >= OCR_INTERVAL:
                last_ocr_submit = now

                # Keep only the newest pending frame.
                try:
                    while True:
                        ocr_queue.get_nowait()
                        ocr_queue.task_done()
                except queue.Empty:
                    pass

                try:
                    ocr_queue.put_nowait(frame.copy())
                except queue.Full:
                    pass

            cv2.putText(
                annotated_frame,
                "Point camera at text",
                (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

        # FPS calculation for the main processing/display loop.
        fps_counter += 1
        now = time.time()

        if now - fps_start >= 1.0:
            inference_fps = fps_counter / (now - fps_start)
            fps_counter = 0
            fps_start = now

        cv2.putText(
            annotated_frame,
            f"MODE: {display_mode}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0) if mode == "OBJECT" else (255, 200, 0),
            2
        )

        cv2.putText(
            annotated_frame,
            f"Processing FPS: {inference_fps:.1f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 255),
            2
        )

        cv2.putText(
            annotated_frame,
            f"Camera FPS: {stream_fps:.1f}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 255),
            2
        )

        cv2.imshow("SMART CAP - OBJECT / OCR MODES", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\nStopping Smart Cap...")

except Exception as error:
    print("Main processing error:", error)

finally:
    print("\nShutting down Smart Cap...")

    stream_stopped.set()
    frame_ready.set()

    try:
        client.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass

    client.close()

    try:
        speech_queue.put(None, timeout=2)
        speech_thread.join(timeout=10)
    except queue.Full:
        print("Speech queue could not be stopped cleanly.")

    cv2.destroyAllWindows()
    print("Smart Cap stopped.")
