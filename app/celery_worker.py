import os
from celery import Celery
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from app.models import Book
from app.db import Base, DATABASE_URL
from storytelling.narrative_engine import FableFactory

redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "kids_story_tasks",
    broker=redis_url,
    backend=redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

# Set up sync SQLAlchemy for Celery
db_url_sync = DATABASE_URL.replace("+asyncpg", "") if "+asyncpg" in DATABASE_URL else DATABASE_URL
engine = create_engine(db_url_sync)
Session = sessionmaker(bind=engine)

@celery_app.task(bind=True)
def generate_book_task(self, book_id, prompt):
    session = Session()
    try:
        # Mark as in_progress
        book = session.execute(select(Book).where(Book.book_id == book_id)).scalar_one_or_none()
        if not book:
            return {"status": "not_found", "book_id": book_id}
        book.status = "in_progress"
        session.commit()
        # Generate story package (using your async FableFactory)
        import asyncio
        factory = FableFactory()
        story_package = asyncio.run(factory.generate_story_package(prompt))
        # Update book with result
        book.status = "ready"
        book.result = story_package
        session.commit()
        return {"status": "ready", "book_id": book_id}
    except Exception as e:
        session.rollback()
        # Mark as failed
        book = session.execute(select(Book).where(Book.book_id == book_id)).scalar_one_or_none()
        if book:
            book.status = "failed"
            session.commit()
        return {"status": "failed", "book_id": book_id, "error": str(e)}
    finally:
        session.close()
