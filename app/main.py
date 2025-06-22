import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

from dotenv import load_dotenv

load_dotenv()

from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.celery_worker import generate_book_task
from app.crud import (
    create_book,
    create_translated_pages,
    fetch_available_languages_for_book,
    fetch_book_by_id,
    fetch_book_by_request_id,
    fetch_page_by_book_order_language,
    fetch_pages_by_book_id_and_language,
)
from app.db import SessionLocal
from app.models import SupportedLanguage
from app.model_providers import ModelPreferences, TextModel, ImageModel, AudioModel, ModelConfig
from app.translation_service import TranslationService
from app.utils import log_memory_usage

logger = logging.getLogger("kids-story-app")

app = FastAPI()


async def get_db():
    async with SessionLocal() as session:
        yield session


# --- Data models ---
class ModelPreferencesRequest(BaseModel):
    """Model preferences for story generation"""
    text_model: Optional[str] = ModelConfig.DEFAULT_TEXT_MODEL.value
    image_model: Optional[str] = ModelConfig.DEFAULT_IMAGE_MODEL.value  
    audio_model: Optional[str] = ModelConfig.DEFAULT_AUDIO_MODEL.value


class BookCreateRequest(BaseModel):
    prompt: str
    request_id: str
    target_languages: Optional[List[str]] = (
        None  # List of language codes like ["es", "zh", "fr"]
    )
    difficulty_level: Optional[int] = (
        2  # Age-based difficulty level (0-10), default to Level 2 (Age 6)
    )
    model_preferences: Optional[ModelPreferencesRequest] = None


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
    text_model: Optional[str] = None
    image_model: Optional[str] = None  
    audio_model: Optional[str] = None
    creation_duration: Optional[float] = None
    correct_first_try: Optional[bool] = None


# --- Redis-based status tracking ---
BOOK_PREFIX = "book:"


# --- FastAPI endpoints ---
@app.post("/books/", response_model=BookResponse, status_code=202)
async def create_book_endpoint(
    req: BookCreateRequest, db: AsyncSession = Depends(get_db)
):
    log_memory_usage("main.create_book_endpoint: start")
    # Check idempotency by request_id
    existing = await fetch_book_by_request_id(db, req.request_id)
    if existing:
        pages = await fetch_pages_by_book_id_and_language(
            db, existing.book_id, SupportedLanguage.ENGLISH
        )
        available_languages = await fetch_available_languages_for_book(
            db, existing.book_id
        )
        log_memory_usage("main.create_book_endpoint: returning existing")
        return BookResponse(
            book_id=str(existing.book_id),
            status=existing.status,
            title=existing.title,
            pages=[
                PageResponse(
                    order=p.order,
                    text=p.text,
                    image_prompt=p.image_prompt,
                    audio_url=p.audio_url,
                    image_url=p.image_url,
                    language=p.language.value,
                )
                for p in pages
            ],
            available_languages=available_languages,
            difficulty_level=existing.difficulty_level,
            calculated_readability_score=existing.calculated_readability_score,
            text_model=existing.text_model,
            image_model=existing.image_model,
            audio_model=existing.audio_model,
            creation_duration=existing.creation_duration,
            correct_first_try=existing.correct_first_try,
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
                raise HTTPException(
                    status_code=400, detail=f"Unsupported language: {lang_code}"
                )

    # Validate and prepare model preferences
    model_preferences_dict = None
    if req.model_preferences:
        try:
            # Validate model selections
            TextModel(req.model_preferences.text_model)
            ImageModel(req.model_preferences.image_model)
            AudioModel(req.model_preferences.audio_model)
            
            model_preferences_dict = {
                "text_model": req.model_preferences.text_model,
                "image_model": req.model_preferences.image_model,
                "audio_model": req.model_preferences.audio_model,
            }
        except ValueError as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid model selection: {str(e)}"
            )

    # Enqueue Celery task with target languages and model preferences
    generate_book_task.delay(
        str(book.book_id),
        req.prompt,
        target_languages if target_languages else None,
        req.difficulty_level,
        model_preferences_dict,
    )
    log_memory_usage("main.create_book_endpoint: after enqueue")
    return BookResponse(
        book_id=str(book.book_id),
        status=book.status,
        title=book.title,
        pages=[],
        difficulty_level=book.difficulty_level,
        calculated_readability_score=book.calculated_readability_score,
        text_model=book.text_model,
        image_model=book.image_model,
        audio_model=book.audio_model,
        creation_duration=book.creation_duration,
        correct_first_try=book.correct_first_try,
    )


@app.get("/books/{book_id}", response_model=BookResponse)
async def get_book_endpoint(
    book_id: str, language: Optional[str] = "en", db: AsyncSession = Depends(get_db)
):
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
        pages=[
            PageResponse(
                order=p.order,
                text=p.text,
                image_prompt=p.image_prompt,
                audio_url=p.audio_url,
                image_url=p.image_url,
                language=p.language.value,
            )
            for p in pages
        ],
        available_languages=available_languages,
        difficulty_level=book.difficulty_level,
        calculated_readability_score=book.calculated_readability_score,
        text_model=book.text_model,
        image_model=book.image_model,
        audio_model=book.audio_model,
        creation_duration=book.creation_duration,
        correct_first_try=book.correct_first_try,
    )


# Saved for future use, not currently used
@app.post("/books/{book_id}/translate")
async def translate_book_endpoint(
    book_id: str, req: TranslationRequest, db: AsyncSession = Depends(get_db)
):
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
            raise HTTPException(
                status_code=400, detail=f"Unsupported language: {lang_code}"
            )

    if not target_languages:
        raise HTTPException(
            status_code=400, detail="No valid target languages provided"
        )

    # Get English pages
    english_pages = await fetch_pages_by_book_id_and_language(
        db, book.book_id, SupportedLanguage.ENGLISH
    )
    if not english_pages:
        raise HTTPException(
            status_code=400, detail="No English pages found for translation"
        )

    # Start translation process
    translation_service = TranslationService()

    try:
        # Convert pages to dict format for translation service
        pages_data = [
            {"text": p.text, "imagePrompt": p.image_prompt} for p in english_pages
        ]

        # Translate to each target language
        for target_language in target_languages:
            translated_pages = await translation_service.translate_story_pages(
                pages_data, target_language
            )

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
                    language=target_language,
                )

        await db.commit()
        log_memory_usage("main.translate_book_endpoint: after translation")

        return {
            "message": f"Translation started for languages: {[lang.value for lang in target_languages]}",
            "book_id": book_id,
        }

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


class AudioGenerationRequest(BaseModel):
    language: str = "en"

@app.post("/books/{book_id}/pages/{page_order}/audio")
async def generate_page_audio_endpoint(
    book_id: str,
    page_order: int,
    request: AudioGenerationRequest,
    db: AsyncSession = Depends(get_db),
):
    """Generate audio for a specific page on-demand."""
    log_memory_usage("main.generate_page_audio_endpoint: start")
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
        lang_enum = SupportedLanguage(request.language)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {request.language}")

    # Get the specific page
    page = await fetch_page_by_book_order_language(
        db, book.book_id, page_order, lang_enum
    )
    if not page:
        raise HTTPException(status_code=404, detail="Page not found")

    # Check if audio already exists
    if page.audio_url:
        return {"audio_url": page.audio_url, "cached": True}

    # Generate audio on-demand
    try:
        from app.narrative_engine import FableFactory
        from app.model_providers import ModelPreferences

        # Audio model is controlled by backend defaults - no user selection
        logger.info(f"Audio generation request - Language: {request.language}")
        preferences = ModelPreferences()  # Always use backend defaults for audio
        logger.info(f"Using backend default audio model: {preferences.audio_model.value}")
            
        factory = FableFactory(preferences)

        # Select appropriate voice for the language
        voice = factory._select_voice_for_language(request.language)
        logger.info(
            f"Generating audio for page {page_order} in language '{request.language}' using voice '{voice}' with model '{preferences.audio_model.value}'"
        )

        # Generate audio for this specific page
        audio_result = await factory._generate_single_narration(
            page.text, page_order, voice
        )

        if audio_result and audio_result.get("audio_url"):
            # Update the page with the new audio URL
            page.audio_url = audio_result["audio_url"]
            await db.commit()

            log_memory_usage("main.generate_page_audio_endpoint: success")
            return {"audio_url": audio_result["audio_url"], "cached": False}
        else:
            raise HTTPException(status_code=500, detail="Failed to generate audio")

    except Exception as e:
        await db.rollback()
        logger.error(f"Error generating audio for page {page_order}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Audio generation failed: {str(e)}"
        )


@app.get("/models")
def get_available_models():
    """Get available AI models for text, image, and audio generation"""
    return {
        "text_models": [
            {
                "id": model.value,
                "name": ModelConfig.TEXT_MODEL_NAMES[model],
                "provider": ModelConfig.TEXT_PROVIDERS[model]
            }
            for model in TextModel
        ],
        "image_models": [
            {
                "id": model.value,
                "name": ModelConfig.IMAGE_MODEL_NAMES[model],
                "provider": ModelConfig.IMAGE_PROVIDERS[model]
            }
            for model in ImageModel
        ],
        "audio_models": [
            {
                "id": model.value,
                "name": ModelConfig.AUDIO_MODEL_NAMES[model],
                "provider": ModelConfig.AUDIO_PROVIDERS[model]
            }
            for model in AudioModel
        ],
        "defaults": {
            "text_model": ModelConfig.DEFAULT_TEXT_MODEL.value,
            "image_model": ModelConfig.DEFAULT_IMAGE_MODEL.value,
            "audio_model": ModelConfig.DEFAULT_AUDIO_MODEL.value,
        }
    }


@app.get("/")
def read_root():
    return {"message": "Kids Story FastAPI app is running!"}


# --- Utility functions for Celery worker ---
def update_book_and_scenes(book_id: str, story_package: dict):
    import json

    redis_client.hset(
        f"{BOOK_PREFIX}{book_id}",
        mapping={"status": "ready", "book": json.dumps(story_package)},
    )


def mark_book_failed(book_id: str):
    redis_client.hset(f"{BOOK_PREFIX}{book_id}", "status", "failed")


def generate_story_package(prompt: str) -> dict:
    # Placeholder for actual generation logic
    import time

    time.sleep(1)  # Simulate work (replace with real logic)
    return {"title": "Generated Story", "content": f"Story for prompt: {prompt}"}
