#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import socket
import struct
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from segmentation_protocol import (
    MSG_CONFIGURE,
    MSG_ERROR,
    MSG_SEGMENT,
    decode_message_type,
    encode_configure_ack,
    encode_error_response,
    encode_segment_result,
)


args = argparse.ArgumentParser(description="Segmentation Server")
args.add_argument("--host", type=str, default="127.0.0.1")
args.add_argument("--port", type=int, default=8765)
args.add_argument(
    "--method",
    type=str,
    default="sam3",
    choices=["sam3"],
    help="Segmentation method to use",
)
parsed_args = args.parse_args()

if parsed_args.method == "sam3":
    from inference.sam3 import configure_prompts, segment_image
else:
    raise ValueError(f"Unsupported segmentation method: {parsed_args.method}")


HOST = parsed_args.host
PORT = parsed_args.port


def recvall(sock: socket.socket, n: int) -> bytes | None:
    data = b""
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


def recv_msg(sock: socket.socket) -> bytes | None:
    raw_len = recvall(sock, 4)
    if not raw_len:
        return None
    msg_len = struct.unpack(">I", raw_len)[0]
    return recvall(sock, msg_len)


def send_msg(sock: socket.socket, data_bytes: bytes) -> None:
    msg_len = struct.pack(">I", len(data_bytes))
    sock.sendall(msg_len + data_bytes)


def _decode_prompts(request: np.lib.npyio.NpzFile) -> tuple[list[str], np.ndarray]:
    if "prompts" not in request or "class_ids" not in request:
        raise RuntimeError("El mensaje configure debe incluir 'prompts' y 'class_ids'.")

    prompts_array = np.asarray(request["prompts"])
    class_ids = np.asarray(request["class_ids"], dtype=np.uint8)

    if prompts_array.ndim != 1:
        raise RuntimeError(f"'prompts' debe ser un vector 1D; recibido {prompts_array.shape}")
    if class_ids.ndim != 1:
        raise RuntimeError(f"'class_ids' debe ser un vector 1D; recibido {class_ids.shape}")
    if prompts_array.shape[0] != class_ids.shape[0]:
        raise RuntimeError("'prompts' y 'class_ids' deben tener la misma longitud.")

    prompts = [str(prompt).strip() for prompt in prompts_array.tolist()]
    if any(not prompt for prompt in prompts):
        raise RuntimeError("La lista de prompts contiene strings vacíos.")

    return prompts, class_ids


def _decode_image(request: np.lib.npyio.NpzFile) -> Image.Image:
    if "image_bgr" not in request:
        raise RuntimeError("El mensaje segment debe incluir 'image_bgr'.")

    image_bgr = np.asarray(request["image_bgr"])
    if image_bgr.dtype != np.uint8:
        image_bgr = image_bgr.astype(np.uint8)

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise RuntimeError(f"'image_bgr' debe tener shape (H, W, 3); recibido {image_bgr.shape}")

    image_rgb = image_bgr[:, :, ::-1]
    return Image.fromarray(image_rgb, mode="RGB")


def handle_client(conn: socket.socket) -> None:
    configured = False

    with conn:
        while True:
            msg = recv_msg(conn)
            if msg is None:
                break

            try:
                request = np.load(io.BytesIO(msg), allow_pickle=False)
                message_type = decode_message_type(request)

                if message_type == MSG_CONFIGURE:
                    prompts, class_ids = _decode_prompts(request)
                    t0 = time.time()
                    configure_prompts(prompts, class_ids)
                    configured = True
                    send_msg(conn, encode_configure_ack(len(prompts)))
                    print(
                        f"[SegServer] Configured {len(prompts)} prompts in "
                        f"{time.time() - t0:.2f} s."
                    )
                    continue

                if message_type == MSG_SEGMENT:
                    if not configured:
                        raise RuntimeError(
                            "El cliente debe enviar un mensaje 'configure' antes de segmentar."
                        )

                    image = _decode_image(request)
                    t0 = time.time()
                    class_map = segment_image(image)
                    class_map = np.asarray(class_map, dtype=np.uint8)
                    send_msg(conn, encode_segment_result(class_map))
                    print(
                        f"[SegServer] Segmentation done in {time.time() - t0:.2f} s. "
                        f"class_map={class_map.shape}"
                    )
                    continue

                if message_type == MSG_ERROR:
                    raise RuntimeError("El cliente ha enviado un mensaje de error inesperado.")

                raise RuntimeError(f"Tipo de mensaje no soportado: {message_type}")
            except Exception as exc:
                error_message = str(exc)
                print(f"[SegServer] Error procesando mensaje: {error_message}")
                send_msg(conn, encode_error_response(error_message))


def main() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        print(f"[SegServer] Listening on {HOST}:{PORT}")

        while True:
            conn, addr = server_socket.accept()
            print(f"[SegServer] Connected by {addr}")
            try:
                handle_client(conn)
            except Exception as exc:
                print(f"[SegServer] Connection error: {exc}")


if __name__ == "__main__":
    main()
