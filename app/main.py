from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from sqlalchemy import select

from app.api.deps import err, require_self_grab_id
from app.api.routes import auth, health, images, ingest
from app.config import get_settings
from app.database import Base, engine, get_db
from app.models import GrabSubject
from app.repository import list_images_for_grab
from app.schemas import ImageListResponse, ImageOut


@asynccontextmanager
async def lifespan(_app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


def create_app() -> FastAPI:
    s = get_settings()
    app = FastAPI(
        title=s.api_title,
        version=s.api_version,
        lifespan=lifespan,
        description=(
            "Grabpic ingests event photos, clusters faces into stable `grab_id`s, "
            "and lets users retrieve their gallery using a selfie as the authenticator."
        ),
    )
    app.include_router(health.router)
    app.include_router(ingest.router)
    app.include_router(auth.router)
    app.include_router(images.router)

    @app.get("/v1/me/images", response_model=ImageListResponse, tags=["Images"])
    def list_images_me(
        grab_id=Depends(require_self_grab_id),
        db=Depends(get_db),
    ) -> ImageListResponse:
        """Convenience alias: same data as `GET /v1/grabs/{grab_id}/images` using only the Bearer token."""
        exists = db.scalar(select(GrabSubject.grab_id).where(GrabSubject.grab_id == grab_id))
        if exists is None:
            raise err("not_found", "Unknown grab_id.", 404)
        rows = list_images_for_grab(db, grab_id)
        return ImageListResponse(
            grab_id=grab_id,
            images=[ImageOut.model_validate(r) for r in rows],
        )

    return app


app = create_app()
