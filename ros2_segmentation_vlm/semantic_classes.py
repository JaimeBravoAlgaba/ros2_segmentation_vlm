from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


Color = Tuple[int, int, int]


@dataclass(frozen=True)
class SemanticClass:
    id: int
    name: str
    color: Color
    traversable: Optional[bool]
    traversability: Optional[float]
    cost: Optional[float]
    attributes: Dict[str, Any]


@dataclass(frozen=True)
class SemanticClasses:
    source_path: Path
    unknown_class_id: int
    unknown_color: Color
    classes: Tuple[SemanticClass, ...]
    class_names: Tuple[str, ...]
    prompt_class_ids: Tuple[int, ...]
    id_to_color: Dict[int, Color]
    id_to_traversability: Dict[int, Optional[float]]
    id_to_cost: Dict[int, Optional[float]]
    id_to_traversable: Dict[int, Optional[bool]]
    color_lut: np.ndarray
    traversable_lut: np.ndarray
    traversability_lut: np.ndarray
    cost_lut: np.ndarray


def _require_dict(value: Any, field_name: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"'{field_name}' debe ser un objeto JSON.")
    return value


def _parse_uint8(value: Any, field_name: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"'{field_name}' debe ser un entero.")
    if value < 0 or value > 255:
        raise ValueError(f"'{field_name}' debe estar en [0, 255].")
    return value


def _parse_color(value: Any, field_name: str) -> Color:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"'{field_name}' debe ser una lista RGB de 3 enteros.")

    channels: List[int] = []
    for idx, channel in enumerate(value):
        channels.append(_parse_uint8(channel, f"{field_name}[{idx}]"))
    return (channels[0], channels[1], channels[2])


def _parse_optional_bool(value: Any, field_name: str) -> Optional[bool]:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"'{field_name}' debe ser booleano si se define.")
    return value


def _parse_optional_float(value: Any, field_name: str) -> Optional[float]:
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise ValueError(f"'{field_name}' debe ser numérico si se define.")
    return float(value)


def load_semantic_classes(path: str) -> SemanticClasses:
    json_path = Path(path).expanduser().resolve()
    if not json_path.is_file():
        raise FileNotFoundError(f"No existe el fichero de clases semánticas: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        root = json.load(f)

    root = _require_dict(root, "root")

    unknown_class_id = _parse_uint8(root.get("unknown_class_id", 255), "unknown_class_id")
    unknown_color = _parse_color(root.get("unknown_color", [0, 0, 0]), "unknown_color")

    classes_raw = root.get("classes")
    if not isinstance(classes_raw, list) or len(classes_raw) == 0:
        raise ValueError("'classes' debe ser una lista no vacía.")

    parsed_classes: List[SemanticClass] = []
    seen_ids = set()
    seen_names = set()

    for idx, item in enumerate(classes_raw):
        entry = _require_dict(item, f"classes[{idx}]")

        class_id = _parse_uint8(entry.get("id"), f"classes[{idx}].id")
        if class_id == unknown_class_id:
            raise ValueError(
                f"classes[{idx}].id={class_id} colisiona con unknown_class_id={unknown_class_id}."
            )
        if class_id in seen_ids:
            raise ValueError(f"ID de clase duplicado: {class_id}")
        seen_ids.add(class_id)

        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"classes[{idx}].name debe ser un string no vacío.")
        name = name.strip()
        if name in seen_names:
            raise ValueError(f"Nombre de clase duplicado: '{name}'")
        seen_names.add(name)

        color = _parse_color(entry.get("color"), f"classes[{idx}].color")
        traversable = _parse_optional_bool(entry.get("traversable"), f"classes[{idx}].traversable")
        traversability = _parse_optional_float(
            entry.get("traversability"),
            f"classes[{idx}].traversability",
        )
        cost = _parse_optional_float(entry.get("cost"), f"classes[{idx}].cost")

        if traversability is not None and not (0.0 <= traversability <= 1.0):
            raise ValueError(
                f"classes[{idx}].traversability debe estar en [0.0, 1.0]."
            )
        if cost is not None and cost < 0.0:
            raise ValueError(f"classes[{idx}].cost debe ser >= 0.0.")

        attributes = {
            key: value
            for key, value in entry.items()
            if key not in {"id", "name", "color", "traversable", "traversability", "cost"}
        }

        parsed_classes.append(
            SemanticClass(
                id=class_id,
                name=name,
                color=color,
                traversable=traversable,
                traversability=traversability,
                cost=cost,
                attributes=attributes,
            )
        )

    if len(parsed_classes) > 255:
        raise ValueError(
            "Hay más de 255 clases configuradas. El class_map uint8 no puede representarlas."
        )

    ordered_classes = tuple(sorted(parsed_classes, key=lambda item: item.id))
    class_names = tuple(cls.name for cls in ordered_classes)
    prompt_class_ids = tuple(cls.id for cls in ordered_classes)
    id_to_color = {cls.id: cls.color for cls in ordered_classes}
    id_to_traversability = {cls.id: cls.traversability for cls in ordered_classes}
    id_to_cost = {cls.id: cls.cost for cls in ordered_classes}
    id_to_traversable = {cls.id: cls.traversable for cls in ordered_classes}

    color_lut = np.tile(np.asarray(unknown_color, dtype=np.uint8), (256, 1))
    traversable_lut = np.zeros((256,), dtype=np.uint8)
    traversability_lut = np.full((256,), np.nan, dtype=np.float32)
    cost_lut = np.ones((256,), dtype=np.float32)

    for cls in ordered_classes:
        color_lut[cls.id] = np.asarray(cls.color, dtype=np.uint8)
        traversable_lut[cls.id] = np.uint8(1 if cls.traversable else 0)
        traversability_lut[cls.id] = np.float32(
            cls.traversability if cls.traversability is not None else 0.0
        )
        cost_lut[cls.id] = np.float32(cls.cost if cls.cost is not None else 1.0)

    return SemanticClasses(
        source_path=json_path,
        unknown_class_id=unknown_class_id,
        unknown_color=unknown_color,
        classes=ordered_classes,
        class_names=class_names,
        prompt_class_ids=prompt_class_ids,
        id_to_color=id_to_color,
        id_to_traversability=id_to_traversability,
        id_to_cost=id_to_cost,
        id_to_traversable=id_to_traversable,
        color_lut=color_lut,
        traversable_lut=traversable_lut,
        traversability_lut=traversability_lut,
        cost_lut=cost_lut,
    )


def colorize_class_map(class_map: np.ndarray, semantic_classes: SemanticClasses) -> np.ndarray:
    class_map = np.asarray(class_map, dtype=np.uint8)
    if class_map.ndim != 2:
        raise ValueError(f"class_map debe tener shape (H, W); recibido {class_map.shape}")
    return semantic_classes.color_lut[class_map]
