#!/usr/bin/env python3
import socket
import struct
import io

import numpy as np
from PIL import Image

import argparse
args = argparse.ArgumentParser(description="Segmentation Server")
args.add_argument("--host", type=str, default="127.0.0.1")
args.add_argument("--port", type=int, default=8765)
args.add_argument("--method", type=str, default="sam3", choices=["gdinosam2", "clipseg", "sam3"], help="Segmentation method to use")
parsed_args = args.parse_args()

if parsed_args.method == "gdinosam2":
    from inference.gdinosam2 import segment_image
elif parsed_args.method == "clipseg":
    from inferece.clipseg import segment_image
elif parsed_args.method == "sam3":
    from inference.sam3 import segment_image
else:
    raise ValueError(f"Unsupported segmentation method: {parsed_args.method}")

HOST = parsed_args.host
PORT = parsed_args.port

def recvall(sock, n):
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


def recv_msg(sock):
    raw_len = recvall(sock, 4)
    if not raw_len:
        return None
    msg_len = struct.unpack('>I', raw_len)[0]
    return recvall(sock, msg_len)


def send_msg(sock, data_bytes):
    msg_len = struct.pack('>I', len(data_bytes))
    sock.sendall(msg_len + data_bytes)


def handle_client(conn):
    with conn:
        while True:
            msg = recv_msg(conn)
            if msg is None:
                break

            # --- 1) Deserialize ndarray using numpy.load (ROS -> server) ---
            buf = io.BytesIO(msg)
            img = np.load(buf, allow_pickle=False)  # BGR uint8 HxWx3
            print(f"[INFO] Loaded input, shape={img.shape}, dtype={img.dtype}, len={len(msg)} bytes")

            # --- 2) Convert np.ndarray -> PIL.Image (RGB) for segment_image ---
            if isinstance(img, np.ndarray):
                if img.ndim == 3 and img.shape[2] == 3:
                    rgb = img[:, :, ::-1]   # BGR -> RGB
                else:
                    rgb = img
                pil_img = Image.fromarray(rgb)
            elif isinstance(img, Image.Image):
                pil_img = img
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")

            # --- 3) Run segmentation ---
            seg_out = segment_image(pil_img)

            # --- 4) Back to numpy RGB array ---
            if isinstance(seg_out, Image.Image):
                seg_rgb = np.array(seg_out)
            else:
                seg_rgb = np.asarray(seg_out)

            seg_rgb = seg_rgb.astype(np.uint8)

            # If the model returns RGBA (4 channels), drop alpha
            if seg_rgb.ndim == 3 and seg_rgb.shape[2] == 4:
                seg_rgb = seg_rgb[:, :, :3]  # keep RGB only

            h, w, c = seg_rgb.shape
            print(f"[INFO] Seg result shape={seg_rgb.shape}, sending back...")

            # --- 5) Encode as: [H,W,C header][raw bytes] ---
            header = struct.pack('>III', h, w, c)
            payload = header + seg_rgb.tobytes()

            send_msg(conn, payload)
            print(f"[SegServer] Sent segmented image with size={seg_rgb.shape}.")


def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"[SegServer] Listening on {HOST}:{PORT}")

        while True:
            conn, addr = s.accept()
            print(f"[SegServer] Connected by {addr}")
            try:
                handle_client(conn)
            except Exception as e:
                print(f"[SegServer] Error: {e}")


if __name__ == '__main__':
    main()
