
import cv2
import socket
import struct
import numpy as np
import time

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


client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

print("Connecting to Raspberry Pi...")

try:
    client.connect((PI_IP, PORT))
    print("Connected to Raspberry Pi!")
    print("Receiving live camera feed...")
    print("Press Q to quit.")

    frame_count = 0
    start_time = time.time()
    fps = 0

    while True:
        # Receive frame size
        header = receive_exact(client, 4)

        if header is None:
            print("Connection closed by Raspberry Pi")
            break

        frame_size = struct.unpack("!I", header)[0]

        # Receive JPEG image
        frame_data = receive_exact(client, frame_size)

        if frame_data is None:
            print("Incomplete frame received")
            break

        # Decode image
        frame = cv2.imdecode(
            np.frombuffer(frame_data, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )

        if frame is None:
            continue

        # Measure resolution
        height, width = frame.shape[:2]

        # Calculate FPS
        frame_count += 1
        elapsed = time.time() - start_time

        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            start_time = time.time()

            print(
                f"Received resolution: {width}x{height} | "
                f"FPS: {fps:.1f}"
            )

        # Display resolution and FPS
        cv2.putText(
            frame,
            f"Resolution: {width}x{height}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow("Smart Cap - Live Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except ConnectionRefusedError:
    print("Connection refused. Check whether the Pi camera server is running.")

except Exception as error:
    print("Error:", error)

finally:
    client.close()
    cv2.destroyAllWindows()
    print("Camera receiver stopped.")

