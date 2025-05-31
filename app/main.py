import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from app.db import SessionLocal
from app.crud import create_book, fetch_book_by_id, fetch_book_by_request_id
from app.celery_worker import generate_book_task
from app.utils import log_memory_usage

app = FastAPI()

async def get_db():
    async with SessionLocal() as session:
        yield session

# --- Data models ---
class BookCreateRequest(BaseModel):
    prompt: str
    request_id: str

class PageResponse(BaseModel):
    order: int
    text: str
    image_prompt: Optional[str] = None
    audio_url: Optional[str] = None
    image_url: Optional[str] = None

class BookResponse(BaseModel):
    book_id: str
    status: str
    title: Optional[str] = None
    pages: List[PageResponse] = []

# --- Redis-based status tracking ---
BOOK_PREFIX = "book:"

# --- FastAPI endpoints ---
@app.post("/books/", response_model=BookResponse, status_code=202)
async def create_book_endpoint(req: BookCreateRequest, db: AsyncSession = Depends(get_db)):
    log_memory_usage("main.create_book_endpoint: start")
    # Check idempotency by request_id
    existing = await fetch_book_by_request_id(db, req.request_id)
    if existing:
        from app.crud import fetch_pages_by_book_id
        pages = await fetch_pages_by_book_id(db, existing.book_id)
        log_memory_usage("main.create_book_endpoint: returning existing")
        return BookResponse(
            book_id=str(existing.book_id),
            status=existing.status,
            title=existing.title,
            pages=[PageResponse(order=p.order, text=p.text, image_prompt=p.image_prompt, audio_url=p.audio_url, image_url=p.image_url) for p in pages]
        )
    # Otherwise, create a new book
    book = await create_book(db, req.prompt, req.request_id)
    # Enqueue Celery task
    generate_book_task.delay(str(book.book_id), req.prompt)
    log_memory_usage("main.create_book_endpoint: after enqueue")
    return BookResponse(book_id=str(book.book_id), status=book.status, title=book.title, pages=[])

@app.get("/books/{book_id}", response_model=BookResponse)
async def get_book_endpoint(book_id: str, db: AsyncSession = Depends(get_db)):
    log_memory_usage("main.get_book_endpoint: start")
    import uuid as uuid_mod
    try:
        book_uuid = uuid_mod.UUID(book_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid book_id format")
    book = await fetch_book_by_id(db, book_uuid)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    from app.crud import fetch_pages_by_book_id
    pages = await fetch_pages_by_book_id(db, book.book_id)
    log_memory_usage("main.get_book_endpoint: after fetch")
    return BookResponse(
        book_id=book_id,
        status=book.status,
        title=book.title,
        pages=[PageResponse(order=p.order, text=p.text, image_prompt=p.image_prompt, audio_url=p.audio_url, image_url=p.image_url) for p in pages]
    )

@app.get("/")
def read_root():
    return {"message": "Kids Story FastAPI app is running!"}

# --- Utility functions for Celery worker ---
def update_book_and_scenes(book_id: str, story_package: dict):
    import json
    redis_client.hset(f"{BOOK_PREFIX}{book_id}", mapping={
        "status": "ready",
        "book": json.dumps(story_package)
    })

def mark_book_failed(book_id: str):
    redis_client.hset(f"{BOOK_PREFIX}{book_id}", "status", "failed")

def generate_story_package(prompt: str) -> dict:
    # Placeholder for actual generation logic
    import time
    time.sleep(1)  # Simulate work (replace with real logic)
    return {"title": "Generated Story", "content": f"Story for prompt: {prompt}"}
