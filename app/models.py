import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func
from app.db import Base
import uuid

class Book(Base):
    __tablename__ = "books"

    book_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_id: Mapped[str] = mapped_column(sa.String, unique=True, nullable=False)
    prompt: Mapped[str] = mapped_column(sa.Text, nullable=False)
    status: Mapped[str] = mapped_column(sa.String(32), nullable=False, default="pending")
    result: Mapped[dict] = mapped_column(sa.JSON, nullable=True)
    created_at: Mapped = mapped_column(sa.DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped = mapped_column(sa.DateTime(timezone=True), onupdate=func.now())
