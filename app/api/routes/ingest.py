from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.orm import Session

from app.api.deps import err
from app.config import Settings, get_settings
from app.database import get_db
from app.schemas import CrawlRequest, CrawlResult, IngestImageResult
from app.services.ingestion import crawl_storage, ingest_upload

router = APIRouter(prefix="/v1/ingest", tags=["Ingestion"])


@router.post("/crawl", response_model=CrawlResult)
def ingest_crawl(
    body: CrawlRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> CrawlResult:
    stats = crawl_storage(db, settings, recursive=body.recursive)
    db.commit()
    return CrawlResult(
        scanned_files=stats.scanned_files,
        processed_images=stats.processed_images,
        faces_indexed=stats.faces_indexed,
        new_grab_ids=stats.new_grab_ids,
        skipped=stats.skipped,
        errors=stats.errors[:200],
    )


@router.post("/image", response_model=IngestImageResult)
async def ingest_image(
    file: UploadFile = File(..., description="Raw image bytes to index."),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> IngestImageResult:
    data = await file.read()
    if not data:
        raise err("empty_file", "Uploaded file is empty.", 400)
    name = file.filename or "upload.jpg"
    rel, img_id, grab_ids, _new_subj, face_count, errs = ingest_upload(db, data, name, settings)
    db.commit()
    if img_id is None or face_count == 0:
        msg = errs[-1] if errs else "No face detected."
        raise err("no_face", msg, 422)
    return IngestImageResult(
        image_id=img_id,
        path=rel,
        grab_ids=grab_ids,
        faces_found=face_count,
    )
