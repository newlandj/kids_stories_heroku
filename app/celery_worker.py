from dotenv import load_dotenv
load_dotenv()

import os
import asyncio
import gc
from celery import Celery
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from app.models import Book, Page
from app.crud import create_pages
from app.db import Base, DATABASE_URL
from app.narrative_engine import FableFactory

redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# Parse the Redis URL to extract components
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode
parsed = urlparse(redis_url)

# Configure Redis SSL settings if using rediss://
redis_use_ssl = parsed.scheme == 'rediss'

# For Heroku Redis, we need to use SSL with specific parameters
if redis_use_ssl:
    # Parse query parameters
    query_params = parse_qs(parsed.query)
    # Add SSL parameters
    query_params['ssl_cert_reqs'] = 'none'  # This is the correct format for redis-py
    query_params['ssl'] = 'true'
    # Rebuild the URL with updated query parameters
    netloc = parsed.netloc.split('@')[-1]  # Remove auth if present
    if parsed.password:
        netloc = f":{parsed.password}@{netloc}"
    if parsed.username:
        netloc = f"{parsed.username}:{netloc}"
    
    redis_url = urlunparse(('rediss', netloc, parsed.path, '', urlencode(query_params, doseq=True), ''))

celery_app = Celery(
    "kids_story_tasks",
    broker=redis_url,
    backend=redis_url,
    broker_use_ssl={'ssl_cert_reqs': None},  # Disable cert verification for Heroku Redis
    redis_backend_use_ssl={'ssl_cert_reqs': None}  # Disable cert verification for Heroku Redis
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    broker_connection_retry_on_startup=True,
    broker_connection_retry=True,
    broker_connection_max_retries=3
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
        factory = FableFactory()
        story_package = asyncio.run(factory.generate_story_package(prompt))
        # Update book with result
        book.status = "ready"
        # Save pages to the DB
        pages = story_package.get("pages")
        if pages:
            for idx, page in enumerate(pages):
                page_obj = Page(
                    book_id=book_id,
                    order=idx,
                    text=page["text"],
                    image_prompt=page.get("imagePrompt"),
                    audio_url=page.get("audioUrl"),
                    image_url=page.get("imageUrl")
                )
                session.add(page_obj)
        session.commit()
        # Explicitly delete large objects and run garbage collection
        del story_package
        del pages
        del factory
        gc.collect()
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
