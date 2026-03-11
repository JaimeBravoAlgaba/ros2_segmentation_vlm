#!/usr/bin/env python3
import socket
import struct
import io

import numpy as np
from PIL import Image
import time

import argparse
args = argparse.ArgumentParser(description="Segmentation Server")
args.add_argument("--host", type=str, default="127.0.0.1")
args.add_argument("--port", type=int, default=8765)
args.add_argument("--method", type=str, default="sam3", choices=["gdinosam2", "clipseg", "sam3"], help="Segmentation method to use")
parsed_args = args.parse_args()

if parsed_args.method == "gdinosam2":
    from inference.gdinosam2 import segment_image
elif parsed_args.method == "clipseg":
    from inference.clipseg import segment_image
elif parsed_args.method == "sam3":
    from inference.sam3 import segment_image
elif parsed_args.method == "sam3_efficient":
    from inference.sam3_efficient.efficientsam3.sam3_efficient import segment_image
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
            t0 = time.time()
            seg_out, class_map = segment_image(pil_img)
            print(f"[INFO] Segmentation done in {time.time() - t0:.2f} seconds.")

            # --- 4) Convert outputs to numpy ---
            if isinstance(seg_out, Image.Image):
                seg_rgb = np.array(seg_out)
            else:
                seg_rgb = np.asarray(seg_out)

            seg_rgb = seg_rgb.astype(np.uint8)
            class_map = np.asarray(class_map).astype(np.uint8)

            # Si viene RGBA, quitar alpha
            if seg_rgb.ndim == 3 and seg_rgb.shape[2] == 4:
                seg_rgb = seg_rgb[:, :, :3]

            print(f"[INFO] seg_rgb shape={seg_rgb.shape}, class_map shape={class_map.shape}")

            # --- 5) Send both arrays packed in one NPZ ---
            out_buf = io.BytesIO()
            np.savez_compressed(
                out_buf,
                seg_rgb=seg_rgb,
                class_map=class_map,
            )
            send_msg(conn, out_buf.getvalue())

            print("[SegServer] Sent segmented image + class map.")


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
