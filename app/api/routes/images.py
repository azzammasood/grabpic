import uuid

from fastapi import APIRouter, Depends, Header
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import err, require_matching_grab
from app.database import get_db
from app.models import GrabSubject
from app.repository import list_images_for_grab
from app.schemas import ImageListResponse, ImageOut

router = APIRouter(prefix="/v1/grabs", tags=["Images"])


@router.get("/{grab_id}/images", response_model=ImageListResponse)
def list_my_images(
    grab_id: uuid.UUID,
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> ImageListResponse:
    require_matching_grab(grab_id, authorization)
    exists = db.scalar(select(GrabSubject.grab_id).where(GrabSubject.grab_id == grab_id))
    if exists is None:
        raise err("not_found", "Unknown grab_id.", 404)
    rows = list_images_for_grab(db, grab_id)
    return ImageListResponse(
        grab_id=grab_id,
        images=[ImageOut.model_validate(r) for r in rows],
    )
