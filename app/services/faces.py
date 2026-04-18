from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from app.config import Settings

logger = logging.getLogger(__name__)


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    if n == 0:
        return vec
    return vec / n


def _facial_area_size(area: dict[str, Any] | None) -> float:
    if not area:
        return 0.0
    w = float(area.get("w", area.get("width", 0)))
    h = float(area.get("h", area.get("height", 0)))
    return w * h


def extract_face_embeddings(
    image_path: str | Path,
    settings: Settings,
) -> list[tuple[np.ndarray, dict[str, Any]]]:
    """
    Returns list of (L2-normalized embedding, facial_area dict) for each detected face.

    DeepFace/TensorFlow are imported on first call so the API can serve `/health`
    immediately after process start; the first face request pays the load cost.
    """
    from deepface import DeepFace

    path = str(image_path)
    detector = settings.detector_backend
    try:
        face_objs = DeepFace.extract_faces(
            path,
            detector_backend=detector,
            enforce_detection=False,
            align=True,
        )
    except Exception as e:
        logger.warning("Primary detector %s failed (%s); falling back to opencv", detector, e)
        face_objs = DeepFace.extract_faces(
            path,
            detector_backend="opencv",
            enforce_detection=False,
            align=True,
        )

    out: list[tuple[np.ndarray, dict[str, Any]]] = []
    if not isinstance(face_objs, list):
        face_objs = [face_objs]

    for item in face_objs:
        if not isinstance(item, dict):
            continue
        face = item.get("face")
        if face is None:
            continue
        face_arr = np.asarray(face, dtype=np.float64)
        if face_arr.max() <= 1.0 and face_arr.min() >= 0:
            face_arr = (face_arr * 255).clip(0, 255).astype(np.uint8)
        else:
            face_arr = face_arr.astype(np.uint8)

        rep = DeepFace.represent(
            face_arr,
            model_name=settings.face_model,
            detector_backend="skip",
            enforce_detection=False,
        )
        emb = np.array(rep[0]["embedding"], dtype=np.float64)
        emb = _l2_normalize(emb)
        facial_area = item.get("facial_area") or {}
        out.append((emb, facial_area if isinstance(facial_area, dict) else {}))
    return out


def extract_embeddings_from_upload(
    data: bytes,
    settings: Settings,
) -> list[tuple[np.ndarray, dict[str, Any]]]:
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".jpg")
    try:
        with open(fd, "wb") as f:
            f.write(data)
        return extract_face_embeddings(path, settings)
    finally:
        Path(path).unlink(missing_ok=True)


def pick_primary_face(
    faces: list[tuple[np.ndarray, dict[str, Any]]],
) -> tuple[np.ndarray, dict[str, Any]] | None:
    if not faces:
        return None
    return max(faces, key=lambda f: _facial_area_size(f[1]))


def best_cosine_match(
    query: np.ndarray,
    candidates: list[tuple[Any, np.ndarray]],
) -> tuple[Any, float] | None:
    if not candidates:
        return None
    q = _l2_normalize(query.astype(np.float64))
    best_id = None
    best_sim = -1.0
    for cid, emb in candidates:
        v = np.asarray(emb, dtype=np.float64)
        v = _l2_normalize(v)
        sim = float(np.dot(q, v))
        if sim > best_sim:
            best_sim = sim
            best_id = cid
    if best_id is None:
        return None
    return best_id, best_sim
