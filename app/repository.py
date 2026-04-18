from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import GrabSubject, ImageGrabMap, StoredImage
from app.services.faces import best_cosine_match


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def list_subject_embeddings(session: Session) -> list[tuple[uuid.UUID, np.ndarray]]:
    rows = session.execute(select(GrabSubject.grab_id, GrabSubject.face_encoding)).all()
    return [(r[0], np.array(r[1], dtype=np.float64)) for r in rows]


def create_subject(session: Session, embedding: list[float]) -> GrabSubject:
    subj = GrabSubject(face_encoding=embedding)
    session.add(subj)
    session.flush()
    return subj


def match_or_create_subject(
    session: Session,
    embedding: np.ndarray,
    threshold: float,
) -> tuple[uuid.UUID, bool]:
    """
    Returns (grab_id, created_new).
    """
    candidates = list_subject_embeddings(session)
    cand_list = [(gid, emb) for gid, emb in candidates]
    match = best_cosine_match(embedding, cand_list)
    if match is not None:
        gid, sim = match
        if sim >= threshold:
            return gid, False
    subj = create_subject(session, embedding.astype(float).tolist())
    return subj.grab_id, True


def upsert_stored_image(session: Session, relative_path: str, file_hash: str | None) -> StoredImage:
    existing = session.scalar(select(StoredImage).where(StoredImage.path == relative_path))
    if existing:
        if file_hash and existing.file_hash != file_hash:
            existing.file_hash = file_hash
        return existing
    img = StoredImage(path=relative_path, file_hash=file_hash)
    session.add(img)
    session.flush()
    return img


def link_image_grab(session: Session, image_id: uuid.UUID, grab_id: uuid.UUID) -> None:
    row = session.scalar(
        select(ImageGrabMap).where(
            ImageGrabMap.image_id == image_id,
            ImageGrabMap.grab_id == grab_id,
        )
    )
    if row:
        return
    session.add(ImageGrabMap(image_id=image_id, grab_id=grab_id))


def list_images_for_grab(session: Session, grab_id: uuid.UUID) -> list[StoredImage]:
    q = (
        select(StoredImage)
        .join(ImageGrabMap, ImageGrabMap.image_id == StoredImage.id)
        .where(ImageGrabMap.grab_id == grab_id)
        .order_by(StoredImage.created_at.desc())
    )
    return list(session.execute(q).scalars().all())


def find_grab_for_selfie(
    session: Session,
    embedding: np.ndarray,
    threshold: float,
) -> tuple[uuid.UUID, float] | None:
    candidates = list_subject_embeddings(session)
    if not candidates:
        return None
    match = best_cosine_match(embedding, candidates)
    if match is None:
        return None
    gid, sim = match
    if sim < threshold:
        return None
    return gid, sim
