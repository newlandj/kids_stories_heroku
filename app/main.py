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
from app.crud import create_book, fetch_book_by_id, fetch_book_by_request_id, fetch_pages_by_book_id, fetch_available_languages_for_book, create_translated_pages, fetch_pages_by_book_id_and_language
from app.celery_worker import generate_book_task
from app.utils import log_memory_usage
from app.models import SupportedLanguage
from app.translation_service import TranslationService

app = FastAPI()

async def get_db():
    async with SessionLocal() as session:
        yield session

# --- Data models ---
class BookCreateRequest(BaseModel):
    prompt: str
    request_id: str
    target_languages: Optional[List[str]] = None  # List of language codes like ["es", "zh", "fr"]
    difficulty_level: Optional[int] = 2  # Age-based difficulty level (0-10), default to Level 2 (Age 6)

class TranslationRequest(BaseModel):
    book_id: str
    target_languages: List[str]

class PageResponse(BaseModel):
    order: int
    text: str
    image_prompt: Optional[str] = None
    audio_url: Optional[str] = None
    image_url: Optional[str] = None
    language: Optional[str] = "en"

class BookResponse(BaseModel):
    book_id: str
    status: str
    title: Optional[str] = None
    pages: List[PageResponse] = []
    available_languages: Optional[List[str]] = None
    difficulty_level: Optional[int] = None
    calculated_readability_score: Optional[float] = None

# --- Redis-based status tracking ---
BOOK_PREFIX = "book:"

# --- FastAPI endpoints ---
@app.post("/books/", response_model=BookResponse, status_code=202)
async def create_book_endpoint(req: BookCreateRequest, db: AsyncSession = Depends(get_db)):
    log_memory_usage("main.create_book_endpoint: start")
    # Check idempotency by request_id
    existing = await fetch_book_by_request_id(db, req.request_id)
    if existing:
        pages = await fetch_pages_by_book_id_and_language(db, existing.book_id, SupportedLanguage.ENGLISH)
        available_languages = await fetch_available_languages_for_book(db, existing.book_id)
        log_memory_usage("main.create_book_endpoint: returning existing")
        return BookResponse(
            book_id=str(existing.book_id),
            status=existing.status,
            title=existing.title,
            pages=[PageResponse(order=p.order, text=p.text, image_prompt=p.image_prompt, audio_url=p.audio_url, image_url=p.image_url, language=p.language.value) for p in pages],
            available_languages=available_languages,
            difficulty_level=existing.difficulty_level,
            calculated_readability_score=existing.calculated_readability_score
        )
    # Otherwise, create a new book
    book = await create_book(db, req.prompt, req.request_id, None, req.difficulty_level)
    
    # Validate target languages if provided
    target_languages = []
    if req.target_languages:
        for lang_code in req.target_languages:
            try:
                SupportedLanguage(lang_code)  # Validate language code
                target_languages.append(lang_code)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Unsupported language: {lang_code}")
    
    # Enqueue Celery task with target languages
    generate_book_task.delay(str(book.book_id), req.prompt, target_languages if target_languages else None, req.difficulty_level)
    log_memory_usage("main.create_book_endpoint: after enqueue")
    return BookResponse(book_id=str(book.book_id), status=book.status, title=book.title, pages=[], difficulty_level=book.difficulty_level, calculated_readability_score=book.calculated_readability_score)

@app.get("/books/{book_id}", response_model=BookResponse)
async def get_book_endpoint(book_id: str, language: Optional[str] = "en", db: AsyncSession = Depends(get_db)):
    log_memory_usage("main.get_book_endpoint: start")
    import uuid as uuid_mod
    try:
        book_uuid = uuid_mod.UUID(book_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid book_id format")
    book = await fetch_book_by_id(db, book_uuid)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    
    # Validate language
    try:
        lang_enum = SupportedLanguage(language)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {language}")
    
    from app.crud import fetch_pages_by_book_id_and_language
    pages = await fetch_pages_by_book_id_and_language(db, book.book_id, lang_enum)
    available_languages = await fetch_available_languages_for_book(db, book.book_id)
    log_memory_usage("main.get_book_endpoint: after fetch")
    return BookResponse(
        book_id=book_id,
        status=book.status,
        title=book.title,
        pages=[PageResponse(order=p.order, text=p.text, image_prompt=p.image_prompt, audio_url=p.audio_url, image_url=p.image_url, language=p.language.value) for p in pages],
        available_languages=available_languages,
        difficulty_level=book.difficulty_level,
        calculated_readability_score=book.calculated_readability_score
    )

@app.post("/books/{book_id}/translate")
async def translate_book_endpoint(book_id: str, req: TranslationRequest, db: AsyncSession = Depends(get_db)):
    """Translate an existing book to additional languages."""
    log_memory_usage("main.translate_book_endpoint: start")
    import uuid as uuid_mod
    try:
        book_uuid = uuid_mod.UUID(book_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid book_id format")
    
    book = await fetch_book_by_id(db, book_uuid)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    
    # Validate target languages
    target_languages = []
    for lang_code in req.target_languages:
        try:
            lang_enum = SupportedLanguage(lang_code)
            if lang_enum != SupportedLanguage.ENGLISH:  # Don't translate to English
                target_languages.append(lang_enum)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Unsupported language: {lang_code}")
    
    if not target_languages:
        raise HTTPException(status_code=400, detail="No valid target languages provided")
    
    # Get English pages
    english_pages = await fetch_pages_by_book_id_and_language(db, book.book_id, SupportedLanguage.ENGLISH)
    if not english_pages:
        raise HTTPException(status_code=400, detail="No English pages found for translation")
    
    # Start translation process
    translation_service = TranslationService()
    
    try:
        # Convert pages to dict format for translation service
        pages_data = [{"text": p.text, "imagePrompt": p.image_prompt} for p in english_pages]
        
        # Translate to each target language
        for target_language in target_languages:
            translated_pages = await translation_service.translate_story_pages(pages_data, target_language)
            
            # Save translated pages to database
            for i, translated_page in enumerate(translated_pages):
                original_page = english_pages[i]
                await create_translated_pages(
                    db=db,
                    book_id=book.book_id,
                    order=original_page.order,
                    text=translated_page["translated_text"],
                    image_prompt=original_page.image_prompt,
                    image_url=original_page.image_url,
                    language=target_language
                )
        
        await db.commit()
        log_memory_usage("main.translate_book_endpoint: after translation")
        
        return {"message": f"Translation started for languages: {[lang.value for lang in target_languages]}", "book_id": book_id}
        
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@app.get("/books/{book_id}/languages")
async def get_book_languages_endpoint(book_id: str, db: AsyncSession = Depends(get_db)):
    """Get available languages for a book."""
    import uuid as uuid_mod
    try:
        book_uuid = uuid_mod.UUID(book_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid book_id format")
    
    book = await fetch_book_by_id(db, book_uuid)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    
    available_languages = await fetch_available_languages_for_book(db, book.book_id)
    return {"book_id": book_id, "available_languages": available_languages}

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
