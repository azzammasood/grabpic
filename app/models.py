import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Double, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class GrabSubject(Base):
    __tablename__ = "grab_subjects"

    grab_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # L2-normalized Facenet512 embedding (512 floats); stored for similarity search.
    face_encoding: Mapped[list[float]] = mapped_column(ARRAY(Double), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    image_links: Mapped[list["ImageGrabMap"]] = relationship(
        back_populates="subject", cascade="all, delete-orphan"
    )


class StoredImage(Base):
    __tablename__ = "stored_images"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    path: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    file_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    grab_links: Mapped[list["ImageGrabMap"]] = relationship(
        back_populates="image", cascade="all, delete-orphan"
    )


class ImageGrabMap(Base):
    __tablename__ = "image_grab_map"

    image_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("stored_images.id", ondelete="CASCADE"), primary_key=True
    )
    grab_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("grab_subjects.grab_id", ondelete="CASCADE"), primary_key=True
    )

    image: Mapped["StoredImage"] = relationship(back_populates="grab_links")
    subject: Mapped["GrabSubject"] = relationship(back_populates="image_links")

