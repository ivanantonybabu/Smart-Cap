
import cv2
import socket
import struct
import numpy as np
import pytesseract
import subprocess
import threading
import queue
import re
import os
import signal
import time

# ==========================================
# CONFIGURATION
# ==========================================

PI_IP = "10.220.139.38"
PORT = 5000

OCR_CONFIGS = [
    "--oem 3 --psm 6",
    "--oem 3 --psm 4",
    "--oem 3 --psm 3"
]

SPEECH_SPEED = "145"

speech_queue = queue.Queue()
ocr_queue = queue.Queue(maxsize=1)

speech_lock = threading.Lock()

speech_process = None
speech_paused = False
stop_event = threading.Event()

# ==========================================
# RECEIVE EXACT DATA FROM RASPBERRY PI
# ==========================================

def receive_exact(sock, size):

    data = bytearray()

    while len(data) < size:

        packet = sock.recv(size - len(data))

        if not packet:
            return None

        data.extend(packet)

    return bytes(data)


# ==========================================
# IMAGE PREPROCESSING
# ==========================================

def preprocess_image(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Enlarge text
    gray = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    # Improve local contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    # Reduce noise
    denoised = cv2.bilateralFilter(
        enhanced,
        7,
        50,
        50
    )

    # Adaptive threshold
    threshold = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    return enhanced, denoised, threshold


# ==========================================
# OCR PROCESSING
# ==========================================

def perform_ocr(frame):

    enhanced, denoised, threshold = preprocess_image(frame)

    cv2.imwrite("ocr_enhanced.jpg", enhanced)
    cv2.imwrite("ocr_threshold.jpg", threshold)

    best_text = ""
    best_score = -1

    for config in OCR_CONFIGS:

        text = pytesseract.image_to_string(
            denoised,
            lang="eng",
            config=config
        ).strip()

        # Simple text-quality score
        words = re.findall(
            r"\b[A-Za-z0-9]+\b",
            text
        )

        score = len(words)

        print(f"\nOCR mode: {config}")
        print("Recognized words:", score)

        if score > best_score:
            best_score = score
            best_text = text

    return best_text, threshold


# ==========================================
# TEXT CLEANING AND SENTENCE SPLITTING
# ==========================================

def split_sentences(text):

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n+",
        " ",
        text
    )

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    cleaned_sentences = []

    for sentence in sentences:

        sentence = sentence.strip()

        if len(sentence) > 2:
            cleaned_sentences.append(sentence)

    return cleaned_sentences


# ==========================================
# SPEECH CONTROL
# ==========================================

def speech_worker():

    global speech_process
    global speech_paused

    while not stop_event.is_set():

        try:
            sentence = speech_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        if sentence is None:
            speech_queue.task_done()
            break

        try:

            with speech_lock:

                if stop_event.is_set():
                    continue

                speech_process = subprocess.Popen(
                    [
                        "espeak-ng",
                        "-s",
                        SPEECH_SPEED,
                        sentence
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )

                if speech_paused:
                    os.kill(
                        speech_process.pid,
                        signal.SIGSTOP
                    )

            speech_process.wait()

        except Exception as error:

            print("Speech error:", error)

        finally:

            with speech_lock:
                speech_process = None

            speech_queue.task_done()


def pause_resume_speech():

    global speech_paused

    with speech_lock:

        if speech_process is None:

            print("No active speech to pause or resume.")
            return

        if not speech_paused:

            os.kill(
                speech_process.pid,
                signal.SIGSTOP
            )

            speech_paused = True
            print("Reading paused.")

        else:

            os.kill(
                speech_process.pid,
                signal.SIGCONT
            )

            speech_paused = False
            print("Reading resumed.")


def cancel_reading():

    global speech_process
    global speech_paused

    # Remove pending sentences
    while True:

        try:
            item = speech_queue.get_nowait()
            speech_queue.task_done()

        except queue.Empty:
            break

    with speech_lock:

        if speech_process is not None:

            speech_process.terminate()
            speech_process = None

        speech_paused = False

    print("Reading cancelled.")


# ==========================================
# OCR WORKER
# ==========================================

def ocr_worker():

    while not stop_event.is_set():

        try:
            frame = ocr_queue.get(timeout=0.2)

        except queue.Empty:
            continue

        if frame is None:
            ocr_queue.task_done()
            break

        try:

            print("\nProcessing captured page...")

            text, processed_image = perform_ocr(frame)

            cv2.imwrite(
                "ocr_sample.jpg",
                frame
            )

            cv2.imshow(
                "OCR Processed Image",
                processed_image
            )

            print("\n========== RECOGNIZED TEXT ==========")
            print(text if text else "No readable text detected")
            print("=====================================")

            if text:

                sentences = split_sentences(text)

                print(
                    f"\nFound {len(sentences)} sentences."
                )

                cancel_reading()

                for sentence in sentences:
                    speech_queue.put(sentence)

                print("Reading started...")

        except Exception as error:

            print("OCR error:", error)

        finally:

            ocr_queue.task_done()


# ==========================================
# START BACKGROUND THREADS
# ==========================================

speech_thread = threading.Thread(
    target=speech_worker,
    daemon=True
)

ocr_thread = threading.Thread(
    target=ocr_worker,
    daemon=True
)

speech_thread.start()
ocr_thread.start()


# ==========================================
# CONNECT TO RASPBERRY PI
# ==========================================

client = socket.socket(
    socket.AF_INET,
    socket.SOCK_STREAM
)

print("\nConnecting to Raspberry Pi...")

try:
    client.connect((PI_IP, PORT))

except ConnectionError as error:

    print("Could not connect to Raspberry Pi:", error)
    stop_event.set()
    raise SystemExit(1)

print("Connected successfully!")

print("\n======================================")
print("       SMART CAP - READING MODE")
print("======================================")
print(" S : Capture page and read")
print(" P : Pause / Resume speech")
print(" C : Cancel current reading")
print(" Q : Quit")
print("======================================\n")


# ==========================================
# MAIN LIVE CAMERA LOOP
# ==========================================

latest_frame = None

try:

    while True:

        header = receive_exact(client, 4)

        if header is None:
            print("Raspberry Pi disconnected.")
            break

        frame_size = struct.unpack(
            "!I",
            header
        )[0]

        # Protect against invalid frame sizes
        if frame_size <= 0 or frame_size > 5_000_000:
            print("Invalid frame received.")
            break

        frame_data = receive_exact(
            client,
            frame_size
        )

        if frame_data is None:
            break

        frame = cv2.imdecode(
            np.frombuffer(
                frame_data,
                dtype=np.uint8
            ),
            cv2.IMREAD_COLOR
        )

        if frame is None:
            continue

        latest_frame = frame.copy()

        display = frame.copy()

        cv2.putText(
            display,
            "S: READ | P: PAUSE/RESUME | C: CANCEL | Q: QUIT",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "SMART CAP - LIVE OCR",
            display
        )

        key = cv2.waitKey(1) & 0xFF

        # CAPTURE PAGE
        if key == ord('s'):

            if latest_frame is not None:

                captured = latest_frame.copy()

                try:

                    ocr_queue.put_nowait(captured)

                    print("\nPage captured!")

                except queue.Full:

                    print(
                        "OCR is busy. Please wait."
                    )

        # PAUSE / RESUME
        elif key == ord('p'):

            pause_resume_speech()

        # CANCEL READING
        elif key == ord('c'):

            cancel_reading()

        # QUIT
        elif key == ord('q'):

            print("\nStopping Smart Cap...")
            break


except KeyboardInterrupt:

    print("\nInterrupted by user.")


finally:

    stop_event.set()

    try:
        client.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass

    client.close()

    cancel_reading()

    try:
        ocr_queue.put_nowait(None)
    except queue.Full:
        pass

    try:
        speech_queue.put_nowait(None)
    except queue.Full:
        pass

    cv2.destroyAllWindows()

    print("Smart Cap reading mode stopped.")
