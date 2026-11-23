#!/usr/bin/env python3
"""
Smart Cap - Acer AI controller
- Receives length-prefixed JPEG frames from Pi TCP/5000.
- Runs YOLO on the RTX GPU.
- Runs PaddleOCR through local OCR worker TCP/6002 in OCR mode.
- Receives OBJECT/OCR mode changes from Pi TCP/6001.
- Sends cancellable Piper WAV audio to Pi TCP/6000.

Audio protocol to Pi:
    b'C'                         -> cancel current playback
    b'A' + uint32_be length + WAV -> play WAV
"""

import cv2
import json
import os
import queue
import socket
import struct
import subprocess
import tempfile
import threading
import time
from collections import defaultdict

import numpy as np
import torch
from ultralytics import YOLO

# -------------------------- USER CONFIGURATION --------------------------
PI_IP = "10.220.139.38"          # Update if the Pi's IP changes
CAMERA_PORT = 5000
AUDIO_PORT = 6000
MODE_PORT = 6001                 # Pi button monitor connects here
OCR_WORKER_HOST = "127.0.0.1"
OCR_WORKER_PORT = 6002

MODEL_PATH = os.path.expanduser("~/smartcap/yolo11s.pt")
PIPER_MODEL = os.path.expanduser("~/smartcap/voices/en_US-lessac-medium.onnx")
PIPER_EXECUTABLE = "piper"

# Performance settings: Pi sends 640x480; inference uses a smaller 416 input.
YOLO_IMAGE_SIZE = 416
YOLO_CONFIDENCE = 0.35
YOLO_MAX_DETECTIONS = 20
DISPLAY_WINDOW = True

# Speech management
TRACK_COOLDOWN_SECONDS = 25.0
CLASS_COOLDOWN_SECONDS = 10.0
ANNOUNCEMENT_BATCH_INTERVAL = 6.0
MAX_OBJECTS_PER_ANNOUNCEMENT = 2
OCR_POLL_INTERVAL = 1.5
OCR_REPEAT_COOLDOWN = 15.0
MIN_OCR_LINE_CONFIDENCE = 0.40

# Colour processing is enabled ONLY for the exact class list supplied by the user.
COLOR_CLASSES = {
    "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "fire hydrant", "stop sign", "parking meter", "bench", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "remote", "microwave", "oven", "toaster",
    "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
}
# -----------------------------------------------------------------------

mode_lock = threading.RLock()
current_mode = "OBJECT"
generation = 0
latest_frame = None
frame_lock = threading.Lock()
frame_sequence = 0
stop_event = threading.Event()

speech_queue = queue.Queue(maxsize=1)
speech_lock = threading.Lock()
active_piper = None
active_piper_lock = threading.Lock()
last_track_spoken = {}
last_class_spoken = defaultdict(float)
last_batch_time = 0.0
last_ocr_text = ""
last_ocr_time = 0.0
last_ocr_request = 0.0
active_ocr_socket = None
active_ocr_lock = threading.Lock()
ocr_request_in_flight = False


def get_generation():
    with mode_lock:
        return generation, current_mode


def receive_exact(sock, size):
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(min(65536, size - len(data)))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def cancel_pi_audio():
    try:
        with socket.create_connection((PI_IP, AUDIO_PORT), timeout=1.0) as s:
            s.settimeout(1.0)
            s.sendall(b"C")
    except OSError:
        pass


def clear_speech_queue():
    while True:
        try:
            speech_queue.get_nowait()
            speech_queue.task_done()
        except queue.Empty:
            break


def change_mode(new_mode):
    global current_mode, generation, last_ocr_text, last_ocr_time, last_ocr_request
    new_mode = new_mode.strip().upper()
    if new_mode not in ("OBJECT", "OCR"):
        return False

    with mode_lock:
        if new_mode == current_mode:
            return True
        current_mode = new_mode
        generation += 1
        current_generation = generation

    print(f"\n[MODE] Switched immediately to {new_mode}; generation={current_generation}")
    clear_speech_queue()

    # Stop Piper generation if it is still running.
    with active_piper_lock:
        proc = active_piper
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass

    # Stop any audio already playing on the Pi.
    cancel_pi_audio()

    # Close an in-flight OCR request socket. OCR inference already running inside
    # PaddleOCR cannot be forcibly interrupted, but its stale response is discarded.
    with active_ocr_lock:
        sock = active_ocr_socket
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass

    last_ocr_text = ""
    last_ocr_time = 0.0
    last_ocr_request = 0.0
    return True


def mode_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", MODE_PORT))
    server.listen(8)
    server.settimeout(1.0)
    print(f"[MODE] Listening for Pi button commands on TCP/{MODE_PORT}")
    while not stop_event.is_set():
        try:
            conn, addr = server.accept()
        except socket.timeout:
            continue
        except OSError:
            break
        with conn:
            try:
                command = conn.recv(32).decode("utf-8", "ignore").strip().upper()
                if change_mode(command):
                    conn.sendall(b"OK")
                else:
                    conn.sendall(b"ER")
            except OSError:
                pass
    server.close()


def camera_receiver():
    global latest_frame, frame_sequence
    while not stop_event.is_set():
        try:
            print(f"[CAMERA] Connecting to Pi {PI_IP}:{CAMERA_PORT} ...")
            with socket.create_connection((PI_IP, CAMERA_PORT), timeout=5.0) as sock:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.settimeout(8.0)
                print("[CAMERA] Connected.")
                while not stop_event.is_set():
                    header = receive_exact(sock, 4)
                    if header is None:
                        raise ConnectionError("Pi camera disconnected")
                    size = struct.unpack("!I", header)[0]
                    if size <= 0 or size > 8 * 1024 * 1024:
                        raise ValueError(f"Invalid JPEG frame size: {size}")
                    payload = receive_exact(sock, size)
                    if payload is None:
                        raise ConnectionError("Incomplete JPEG frame")
                    frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        with frame_lock:
                            latest_frame = frame
                            frame_sequence += 1
        except Exception as exc:
            if not stop_event.is_set():
                print(f"[CAMERA] {exc}; reconnecting in 1 second")
                time.sleep(1)
    print("[CAMERA] Receiver stopped.")


def dominant_color(bgr_crop):
    """Fast approximate colour name from a small crop."""
    if bgr_crop is None or bgr_crop.size == 0:
        return None
    small = cv2.resize(bgr_crop, (32, 32), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)
    # Ignore very dark and nearly white/grey pixels when finding a chromatic hue.
    chroma = pixels[(pixels[:, 1] > 65) & (pixels[:, 2] > 55)]
    if len(chroma) < 12:
        brightness = float(np.mean(pixels[:, 2]))
        if brightness < 55:
            return "black"
        if brightness > 205 and float(np.mean(pixels[:, 1])) < 45:
            return "white"
        return "grey"
    hue = float(np.median(chroma[:, 0]))  # OpenCV hue range 0..179
    sat = float(np.median(chroma[:, 1]))
    val = float(np.median(chroma[:, 2]))
    if val < 65:
        return "dark"
    if hue < 8 or hue >= 170:
        return "red"
    if hue < 20:
        return "orange"
    if hue < 34:
        return "yellow"
    if hue < 78:
        return "green"
    if hue < 100:
        return "cyan"
    if hue < 132:
        return "blue"
    if hue < 155:
        return "purple"
    return "pink"


def get_object_color(frame, xyxy):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return dominant_color(frame[y1:y2, x1:x2])


def queue_speech(text, job_generation):
    if not text.strip():
        return
    with mode_lock:
        if job_generation != generation:
            return
    clear_speech_queue()
    try:
        speech_queue.put_nowait((text.strip(), job_generation))
    except queue.Full:
        pass


def send_wav_to_pi(wav_path, job_generation):
    with mode_lock:
        if job_generation != generation:
            return False
    try:
        size = os.path.getsize(wav_path)
        if size <= 0 or size > 20 * 1024 * 1024:
            print("[AUDIO] WAV size invalid; skipping.")
            return False
        with socket.create_connection((PI_IP, AUDIO_PORT), timeout=3.0) as sock:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(8.0)
            with mode_lock:
                if job_generation != generation:
                    return False
            sock.sendall(b"A" + struct.pack("!I", size))
            with open(wav_path, "rb") as f:
                while True:
                    block = f.read(65536)
                    if not block:
                        break
                    with mode_lock:
                        if job_generation != generation:
                            return False
                    sock.sendall(block)
            response = sock.recv(16)
            return response == b"OK"
    except OSError as exc:
        print(f"[AUDIO] Send failed: {exc}")
        return False


def speech_worker():
    global active_piper
    while not stop_event.is_set():
        try:
            item = speech_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if item is None:
            speech_queue.task_done()
            break
        text, job_generation = item
        try:
            with mode_lock:
                if job_generation != generation:
                    continue
            with tempfile.NamedTemporaryFile(prefix="smartcap_", suffix=".wav", delete=False) as tmp:
                wav_path = tmp.name
            command = [PIPER_EXECUTABLE, "--model", PIPER_MODEL, "--output_file", wav_path]
            with active_piper_lock:
                active_piper = subprocess.Popen(
                    command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, text=True
                )
                proc = active_piper
            try:
                proc.communicate(text, timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
            with active_piper_lock:
                if active_piper is proc:
                    active_piper = None

            with mode_lock:
                still_current = job_generation == generation
            if still_current and os.path.exists(wav_path) and os.path.getsize(wav_path) > 44:
                send_wav_to_pi(wav_path, job_generation)
            try:
                os.remove(wav_path)
            except OSError:
                pass
        except Exception as exc:
            print(f"[SPEECH] {exc}")
        finally:
            speech_queue.task_done()


def object_announcement(detections, now):
    """Return one concise phrase; cooldown is applied per track and per class."""
    global last_batch_time
    if not detections or now - last_batch_time < ANNOUNCEMENT_BATCH_INTERVAL:
        return None

    chosen = []
    seen_classes = set()
    for item in detections:
        cls = item["class"]
        track_id = item["track"]
        track_key = (cls, track_id if track_id is not None else -1)
        if now - last_track_spoken.get(track_key, 0.0) < TRACK_COOLDOWN_SECONDS:
            continue
        if now - last_class_spoken[cls] < CLASS_COOLDOWN_SECONDS:
            continue
        if cls in seen_classes:
            continue
        chosen.append(item)
        seen_classes.add(cls)
        if len(chosen) >= MAX_OBJECTS_PER_ANNOUNCEMENT:
            break

    if not chosen:
        return None

    phrases = []
    for item in chosen:
        if item["color"]:
            phrases.append(f"{item['color']} {item['class']}")
        else:
            phrases.append(item["class"])
        track_key = (item["class"], item["track"] if item["track"] is not None else -1)
        last_track_spoken[track_key] = now
        last_class_spoken[item["class"]] = now

    last_batch_time = now
    if len(phrases) == 1:
        return f"{phrases[0]} detected."
    return f"{phrases[0]} and {phrases[1]} detected."


def run_ocr(frame, job_generation):
    global active_ocr_socket, ocr_request_in_flight
    try:
        h, w = frame.shape[:2]
        scale = min(1.0, 960.0 / max(h, w))
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
        if not ok:
            return
        payload = encoded.tobytes()
        with socket.create_connection((OCR_WORKER_HOST, OCR_WORKER_PORT), timeout=2.0) as sock:
            with active_ocr_lock:
                active_ocr_socket = sock
            sock.settimeout(20.0)
            sock.sendall(struct.pack("!I", len(payload)) + payload)
            header = receive_exact(sock, 4)
            if header is None:
                return
            result_size = struct.unpack("!I", header)[0]
            if result_size <= 0 or result_size > 2 * 1024 * 1024:
                return
            raw = receive_exact(sock, result_size)
            if raw is None:
                return
            with mode_lock:
                if job_generation != generation or current_mode != "OCR":
                    return
            result = json.loads(raw.decode("utf-8", "replace"))
            lines = result.get("lines", [])
            text_lines = []
            for line in lines:
                if float(line.get("confidence", 0.0)) >= MIN_OCR_LINE_CONFIDENCE:
                    value = str(line.get("text", "")).strip()
                    if value:
                        text_lines.append(value)
            recognised = " ".join(text_lines).strip() or str(result.get("text", "")).strip()
            global last_ocr_text, last_ocr_time
            now = time.time()
            if recognised and (recognised != last_ocr_text or now - last_ocr_time >= OCR_REPEAT_COOLDOWN):
                last_ocr_text = recognised
                last_ocr_time = now
                queue_speech(recognised, job_generation)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        if not stop_event.is_set():
            print(f"[OCR] {exc}")
    finally:
        with active_ocr_lock:
            active_ocr_socket = None
            ocr_request_in_flight = False


def main():
    global last_ocr_request, ocr_request_in_flight
    print("[SMARTCAP] Loading YOLO model...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"YOLO model not found: {MODEL_PATH}")
    if not os.path.exists(PIPER_MODEL):
        raise FileNotFoundError(f"Piper model not found: {PIPER_MODEL}")
    model = YOLO(MODEL_PATH)
    use_cuda = torch.cuda.is_available()
    device = 0 if use_cuda else "cpu"
    half = bool(use_cuda)
    if use_cuda:
        print("[GPU]", torch.cuda.get_device_name(0))
    else:
        print("[GPU] CUDA unavailable; using CPU (expect lower FPS)")
    print("[SMARTCAP] Starting services. Initial mode: OBJECT")
    threading.Thread(target=camera_receiver, daemon=True).start()
    threading.Thread(target=mode_server, daemon=True).start()
    threading.Thread(target=speech_worker, daemon=True).start()

    last_sequence = -1
    fps_count = 0
    fps_start = time.time()
    display_fps = 0.0

    try:
        while not stop_event.is_set():
            with frame_lock:
                frame = None if latest_frame is None else latest_frame.copy()
                sequence = frame_sequence
            if frame is None or sequence == last_sequence:
                time.sleep(0.003)
                continue
            last_sequence = sequence
            job_generation, mode = get_generation()
            now = time.time()

            if mode == "OBJECT":
                # Track IDs make cooldowns stable for continuously visible objects.
                try:
                    results = model.track(
                        frame, persist=True, verbose=False, imgsz=YOLO_IMAGE_SIZE,
                        conf=YOLO_CONFIDENCE, max_det=YOLO_MAX_DETECTIONS,
                        device=device, half=half, tracker="bytetrack.yaml"
                    )
                except Exception as exc:
                    print(f"[YOLO] {exc}")
                    time.sleep(0.05)
                    continue

                detected = []
                if results:
                    result = results[0]
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            cls_id = int(box.cls[0].item())
                            cls_name = str(model.names.get(cls_id, cls_id)).lower()
                            conf = float(box.conf[0].item())
                            xyxy = box.xyxy[0].detach().cpu().numpy().tolist()
                            track_id = None
                            if box.id is not None:
                                try:
                                    track_id = int(box.id[0].item())
                                except Exception:
                                    track_id = None
                            color = get_object_color(frame, xyxy) if cls_name in COLOR_CLASSES else None
                            detected.append({
                                "class": cls_name, "confidence": conf,
                                "xyxy": xyxy, "track": track_id, "color": color
                            })
                            if DISPLAY_WINDOW:
                                x1, y1, x2, y2 = [int(v) for v in xyxy]
                                label = f"{cls_name} {conf:.2f}"
                                if color:
                                    label += f" {color}"
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
                                cv2.putText(frame, label, (x1, max(20, y1 - 7)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)

                phrase = object_announcement(detected, now)
                if phrase:
                    queue_speech(phrase, job_generation)
                    print("[OBJECT]", phrase)

            elif mode == "OCR":
                if now - last_ocr_request >= OCR_POLL_INTERVAL:
                    last_ocr_request = now
                    # Never queue multiple OCR frames behind a slow inference.
                    with active_ocr_lock:
                        can_start_ocr = not ocr_request_in_flight
                        if can_start_ocr:
                            ocr_request_in_flight = True
                    if can_start_ocr:
                        threading.Thread(
                            target=run_ocr, args=(frame.copy(), job_generation), daemon=True
                        ).start()
                cv2.putText(frame, "OCR MODE", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 220, 255), 2)

            fps_count += 1
            if now - fps_start >= 1.0:
                display_fps = fps_count / (now - fps_start)
                fps_count = 0
                fps_start = now
            if DISPLAY_WINDOW:
                cv2.putText(frame, f"{mode} | {display_fps:.1f} processed FPS",
                            (12, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 255, 255), 2)
                cv2.imshow("Smart Cap - Acer AI", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        clear_speech_queue()
        cancel_pi_audio()
        if DISPLAY_WINDOW:
            cv2.destroyAllWindows()
        print("[SMARTCAP] Stopped.")


if __name__ == "__main__":
    main()
