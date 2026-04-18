import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class CrawlRequest(BaseModel):
    recursive: bool = Field(
        default=True, description="Walk subdirectories when crawling storage."
    )


class CrawlResult(BaseModel):
    scanned_files: int
    processed_images: int
    faces_indexed: int
    new_grab_ids: int
    skipped: int
    errors: list[str]


class IngestImageResult(BaseModel):
    image_id: uuid.UUID
    path: str
    grab_ids: list[uuid.UUID]
    faces_found: int


class SelfieAuthResponse(BaseModel):
    grab_id: uuid.UUID
    similarity: float = Field(
        ...,
        description="Cosine similarity against the stored embedding (0–1 for L2-normalized vectors).",
    )


class ImageOut(BaseModel):
    id: uuid.UUID
    path: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ImageListResponse(BaseModel):
    grab_id: uuid.UUID
    images: list[ImageOut]
