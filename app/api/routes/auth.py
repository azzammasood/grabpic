from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.orm import Session

from app.api.deps import err
from app.config import Settings, get_settings
from app.database import get_db
from app.repository import find_grab_for_selfie
from app.schemas import SelfieAuthResponse
from app.services.faces import extract_embeddings_from_upload, pick_primary_face

router = APIRouter(prefix="/v1/auth", tags=["Authentication"])


@router.post("/selfie", response_model=SelfieAuthResponse)
async def selfie_auth(
    file: UploadFile = File(..., description="Search token: a face image (selfie)."),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> SelfieAuthResponse:
    data = await file.read()
    if not data:
        raise err("empty_file", "Uploaded file is empty.", 400)
    faces = extract_embeddings_from_upload(data, settings)
    primary = pick_primary_face(faces)
    if primary is None:
        raise err("no_face", "No detectable face in the search token.", 422)
    emb, _area = primary
    match = find_grab_for_selfie(db, emb, settings.match_threshold)
    if match is None:
        raise err(
            "unknown_face",
            "No enrolled subject matches this face at the configured similarity threshold.",
            401,
        )
    grab_id, sim = match
    return SelfieAuthResponse(grab_id=grab_id, similarity=sim)
