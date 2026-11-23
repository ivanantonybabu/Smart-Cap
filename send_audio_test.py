import socket
import struct
import os
import subprocess

PI_IP = "10.220.139.38"
PORT = 6000

VOICE = os.path.expanduser(
    "~/smartcap/voices/en_US-lessac-medium.onnx"
)

AUDIO_FILE = "/tmp/smartcap_transfer_test.wav"

text = (
    "Hello! This is the Smart Cap. "
    "Audio is being generated on the Acer laptop "
    "and transferred to the Raspberry Pi."
)

# Generate speech using Piper
print("Generating speech...")

subprocess.run([
    "python", "-m", "piper",
    "-m", VOICE,
    "-f", AUDIO_FILE,
    "--", text
], check=True)

# Read generated WAV file
with open(AUDIO_FILE, "rb") as f:
    audio_data = f.read()

print("Audio generated successfully.")
print("Connecting to Raspberry Pi...")

# Connect to Pi
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.settimeout(60)

try:
    client.connect((PI_IP, PORT))

    # Send file size followed by audio data
    client.sendall(struct.pack("!I", len(audio_data)))
    client.sendall(audio_data)

    print("Audio sent. Waiting for playback confirmation...")

    response = client.recv(2)

    if response == b"OK":
        print("SUCCESS: Raspberry Pi played the audio!")
    else:
        print("Playback failed:", response)

finally:
    client.close()
