import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func
from app.db import Base
import uuid
import enum

from sqlalchemy.orm import relationship

class SupportedLanguage(enum.Enum):
    ENGLISH = "en"
    SPANISH = "es"
    CHINESE = "zh"
    FRENCH = "fr"
    PORTUGUESE = "pt"

class Book(Base):
    __tablename__ = "books"

    book_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_id: Mapped[str] = mapped_column(sa.String, unique=True, nullable=False)
    prompt: Mapped[str] = mapped_column(sa.Text, nullable=False)
    status: Mapped[str] = mapped_column(sa.String(32), nullable=False, default="pending")
    title: Mapped[str] = mapped_column(sa.String, nullable=True)
    created_at: Mapped[sa.DateTime] = mapped_column(sa.DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[sa.DateTime] = mapped_column(
        sa.DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )

    pages = relationship("Page", back_populates="book", cascade="all, delete-orphan", order_by="Page.order")

class Page(Base):
    __tablename__ = "pages"
    page_id: Mapped[int] = mapped_column(sa.Integer, primary_key=True, autoincrement=True)
    book_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), sa.ForeignKey("books.book_id"), nullable=False)
    order: Mapped[int] = mapped_column(sa.Integer, nullable=False)  # page order in the story
    language: Mapped[SupportedLanguage] = mapped_column(sa.Enum(SupportedLanguage), nullable=False, default=SupportedLanguage.ENGLISH)  # language code
    text: Mapped[str] = mapped_column(sa.Text, nullable=False)
    image_prompt: Mapped[str] = mapped_column(sa.Text, nullable=True)
    image_url: Mapped[str] = mapped_column(sa.Text, nullable=True)  # URL or path to generated image
    audio_url: Mapped[str] = mapped_column(sa.Text, nullable=True)  # URL or path to audio recording

    book = relationship("Book", back_populates="pages")
    
    __table_args__ = (
        sa.UniqueConstraint('book_id', 'order', 'language', name='unique_page_per_language'),
    )
