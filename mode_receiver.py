import socket

HOST = "0.0.0.0"
PORT = 6001

current_mode = "OBJECT"

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

server.bind((HOST, PORT))
server.listen(5)

print("SMART CAP MODE RECEIVER STARTED")
print("Waiting for Raspberry Pi commands...")
print("Initial mode:", current_mode)

try:
    while True:
        conn, address = server.accept()

        try:
            data = conn.recv(1024).decode().strip()

            if data == "OBJECT":
                current_mode = "OBJECT"
                print("\nMODE: OBJECT DETECTION")

            elif data == "OCR":
                current_mode = "OCR"
                print("\nMODE: OCR READING")

            conn.sendall(b"OK")

        except Exception as e:
            print("Error:", e)

        finally:
            conn.close()

except KeyboardInterrupt:
    print("\nMode receiver stopped.")

finally:
    server.close()
