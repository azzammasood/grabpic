from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.orm import Session

from app.config import Settings
from app.repository import link_image_grab, match_or_create_subject, sha256_file, upsert_stored_image
from app.services.faces import extract_face_embeddings

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class CrawlStats:
    scanned_files: int = 0
    processed_images: int = 0
    faces_indexed: int = 0
    new_grab_ids: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)


def iter_image_files(root: Path, recursive: bool) -> list[Path]:
    root = root.resolve()
    if not root.exists():
        return []
    if recursive:
        paths = [p for p in root.rglob("*") if p.is_file()]
    else:
        paths = [p for p in root.iterdir() if p.is_file()]
    return [p for p in paths if p.suffix.lower() in IMAGE_EXTENSIONS]


def process_image_file(
    session: Session,
    absolute_path: Path,
    storage_root: Path,
    settings: Settings,
) -> tuple[uuid.UUID | None, list[uuid.UUID], int, int, list[str]]:
    """
    Returns (image_id, grab_ids, new_grab_ids_count, faces_found, errors).
    If no faces, image_id is None and grab_ids empty.
    """
    errors: list[str] = []
    rel = str(absolute_path.resolve().relative_to(storage_root.resolve()))
    try:
        digest = sha256_file(absolute_path)
    except OSError as e:
        errors.append(f"{rel}: hash failed ({e})")
        return None, [], 0, 0, errors

    faces = extract_face_embeddings(absolute_path, settings)
    if not faces:
        errors.append(f"{rel}: no detectable face")
        return None, [], 0, 0, errors

    img_row = upsert_stored_image(session, rel, digest)
    new_subjects = 0
    grab_ids: list[uuid.UUID] = []
    for emb, _area in faces:
        gid, created = match_or_create_subject(session, emb, settings.match_threshold)
        link_image_grab(session, img_row.id, gid)
        grab_ids.append(gid)
        if created:
            new_subjects += 1
    return img_row.id, grab_ids, new_subjects, len(faces), errors


def ingest_upload(
    session: Session,
    data: bytes,
    original_name: str,
    settings: Settings,
) -> tuple[str, uuid.UUID | None, list[uuid.UUID], int, int, list[str]]:
    """
    Persist bytes under storage/uploads and run the same pipeline as crawl.
    Returns (relative_storage_path, image_id, grab_ids, new_subject_count, faces_found, errors).
    """
    root = Path(settings.storage_path)
    root.mkdir(parents=True, exist_ok=True)
    upload_dir = root / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(original_name).suffix.lower()
    if ext not in IMAGE_EXTENSIONS:
        ext = ".jpg"
    dest = upload_dir / f"{uuid.uuid4().hex}{ext}"
    dest.write_bytes(data)
    rel = str(dest.resolve().relative_to(root.resolve()))
    img_id, gids, new_subj, fc, errs = process_image_file(session, dest, root, settings)
    return rel, img_id, gids, new_subj, fc, errs


def crawl_storage(session: Session, settings: Settings, recursive: bool = True) -> CrawlStats:
    stats = CrawlStats()
    root = Path(settings.storage_path)
    root.mkdir(parents=True, exist_ok=True)
    files = iter_image_files(root, recursive)
    stats.scanned_files = len(files)
    for path in files:
        try:
            _img_id, _gids, new_ids, face_count, errs = process_image_file(
                session, path, root, settings
            )
            stats.errors.extend(errs)
            if face_count == 0:
                stats.skipped += 1
            else:
                stats.processed_images += 1
                stats.faces_indexed += face_count
                stats.new_grab_ids += new_ids
        except Exception as e:
            logger.exception("Failed processing %s", path)
            stats.errors.append(f"{path}: {e}")
    return stats
