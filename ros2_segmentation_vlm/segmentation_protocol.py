from __future__ import annotations

import io
from typing import Iterable, Sequence

import numpy as np


MSG_CONFIGURE = "configure"
MSG_CONFIGURE_ACK = "configure_ack"
MSG_SEGMENT = "segment"
MSG_SEGMENT_RESULT = "segment_result"
MSG_ERROR = "error"


def _to_unicode_array(values: Sequence[str], field_name: str) -> np.ndarray:
    if not values:
        raise ValueError(f"'{field_name}' no puede estar vacío.")
    cleaned = []
    for idx, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"'{field_name}[{idx}]' debe ser un string no vacío.")
        cleaned.append(value.strip())
    return np.asarray(cleaned, dtype=np.str_)


def _to_uint8_array(values: Iterable[int], field_name: str) -> np.ndarray:
    cleaned = []
    for idx, value in enumerate(values):
        if not isinstance(value, (int, np.integer)):
            raise ValueError(f"'{field_name}[{idx}]' debe ser un entero.")
        if value < 0 or value > 255:
            raise ValueError(f"'{field_name}[{idx}]' debe estar en [0, 255].")
        cleaned.append(int(value))
    if not cleaned:
        raise ValueError(f"'{field_name}' no puede estar vacío.")
    return np.asarray(cleaned, dtype=np.uint8)


def encode_configure_request(prompts: Sequence[str], class_ids: Sequence[int]) -> bytes:
    prompts_arr = _to_unicode_array(prompts, "prompts")
    class_ids_arr = _to_uint8_array(class_ids, "class_ids")
    if prompts_arr.shape[0] != class_ids_arr.shape[0]:
        raise ValueError("'prompts' y 'class_ids' deben tener la misma longitud.")

    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        message_type=np.asarray(MSG_CONFIGURE),
        prompts=prompts_arr,
        class_ids=class_ids_arr,
    )
    return buf.getvalue()


def encode_configure_ack(num_prompts: int) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        message_type=np.asarray(MSG_CONFIGURE_ACK),
        num_prompts=np.asarray(num_prompts, dtype=np.int32),
    )
    return buf.getvalue()


def encode_segment_request(image_bgr: np.ndarray) -> bytes:
    image_bgr = np.asarray(image_bgr)
    if image_bgr.dtype != np.uint8:
        raise ValueError("La imagen BGR debe ser uint8.")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"La imagen BGR debe tener shape (H, W, 3); recibido {image_bgr.shape}")

    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        message_type=np.asarray(MSG_SEGMENT),
        image_bgr=image_bgr,
    )
    return buf.getvalue()


def encode_segment_result(class_map: np.ndarray) -> bytes:
    class_map = np.asarray(class_map, dtype=np.uint8)
    if class_map.ndim != 2:
        raise ValueError(f"class_map debe tener shape (H, W); recibido {class_map.shape}")

    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        message_type=np.asarray(MSG_SEGMENT_RESULT),
        class_map=class_map,
    )
    return buf.getvalue()


def encode_error_response(message: str) -> bytes:
    if not isinstance(message, str) or not message:
        raise ValueError("'message' debe ser un string no vacío.")

    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        message_type=np.asarray(MSG_ERROR),
        error=np.asarray(message),
    )
    return buf.getvalue()


def decode_message_type(data: np.lib.npyio.NpzFile) -> str:
    if "message_type" not in data:
        raise ValueError("El mensaje NPZ no contiene 'message_type'.")
    return str(np.asarray(data["message_type"]).item())
