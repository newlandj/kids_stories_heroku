from dotenv import load_dotenv

load_dotenv()

import asyncio
import gc
import os

from celery import Celery
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.db import DATABASE_URL
from app.models import Book, Page, SupportedLanguage
from app.model_providers import ModelPreferences
from app.narrative_engine import FableFactory
from app.readability_analyzer import ReadabilityAnalyzer
from app.utils import log_memory_usage

redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

import ssl

redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
redis_use_ssl = redis_url.startswith("rediss://")

ssl_options = None
if redis_use_ssl:
    ssl_options = {
        "ssl_cert_reqs": ssl.CERT_NONE,
        "ssl_check_hostname": False,
    }

celery_app = Celery(
    "kids_story_tasks",
    broker=redis_url,
    backend=redis_url,
    broker_use_ssl=ssl_options,
    redis_backend_use_ssl=ssl_options,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    broker_connection_retry_on_startup=True,
    broker_connection_retry=True,
    broker_connection_max_retries=3,
)

# Set up sync SQLAlchemy for Celery
db_url_sync = (
    DATABASE_URL.replace("+asyncpg", "") if "+asyncpg" in DATABASE_URL else DATABASE_URL
)
engine = create_engine(db_url_sync)
Session = sessionmaker(bind=engine)


async def generate_media_and_translations_parallel(
    factory, chosen_text_story, prompt, target_language_enums
):
    """Helper function to run media generation and translation in parallel"""
    if target_language_enums:
        # Start translations only - media is already generated in Phase 2
        translation_data = await factory.generate_story_with_translations(
            prompt, target_language_enums, chosen_text_story
        )
        # Combine English story with translations
        story_data = {"en": chosen_text_story}  # Use the story with media from Phase 2
        story_data.update(translation_data)
        return story_data
    else:
        # No translations needed, just return the story with media from Phase 2
        return {"en": chosen_text_story}


@celery_app.task(bind=True)
def generate_book_task(
    self, book_id, prompt, target_languages=None, difficulty_level=None, model_preferences_dict=None
):
    """
    Generate a book with optional multi-language support.

    Args:
        book_id: UUID of the book to generate
        prompt: Story generation prompt
        target_languages: Optional list of language codes to translate to
        difficulty_level: Optional difficulty level for readability analysis
        model_preferences_dict: Optional dictionary of model preferences
    """
    log_memory_usage("celery_worker.generate_book_task: start")
    session = Session()
    try:
        # Mark as in_progress
        book = session.execute(
            select(Book).where(Book.book_id == book_id)
        ).scalar_one_or_none()
        if not book:
            return {"status": "not_found", "book_id": book_id}
        book.status = "in_progress"
        session.commit()
        log_memory_usage("celery_worker.generate_book_task: after mark in_progress")

        # Convert target_languages to enum list if provided
        target_language_enums = []
        if target_languages:
            for lang_code in target_languages:
                try:
                    lang_enum = SupportedLanguage(lang_code)
                    if lang_enum != SupportedLanguage.ENGLISH:
                        target_language_enums.append(lang_enum)
                except ValueError:
                    print(f"Warning: Invalid language code {lang_code}, skipping")

        # Prepare model preferences
        model_preferences = None
        if model_preferences_dict:
            model_preferences = ModelPreferences.from_dict(model_preferences_dict)
            print(f"Book {book_id}: Using model preferences: {model_preferences.to_dict()}")
        else:
            model_preferences = ModelPreferences()  # Use defaults
            print(f"Book {book_id}: Using default model preferences: {model_preferences.to_dict()}")

        # Generate story package with readability checking and translations in parallel
        print(f"Book {book_id}: Creating FableFactory...")
        import time

        factory_start = time.monotonic()
        factory = FableFactory(model_preferences=model_preferences)
        factory_elapsed = time.monotonic() - factory_start
        print(
            f"Book {book_id}: FableFactory creation took {factory_elapsed:.3f} seconds"
        )

        print(f"Book {book_id}: Starting story generation...")
        story_start = time.monotonic()
        story_data = asyncio.run(
            factory.generate_story_with_readability_check_first(
                prompt,
                difficulty_level=difficulty_level,
                target_languages=target_language_enums,
            )
        )
        story_elapsed = time.monotonic() - story_start
        print(f"Book {book_id}: Story generation took {story_elapsed:.3f} seconds")

        log_memory_usage("celery_worker.generate_book_task: story generation complete")

        # Update the book's readability score with final calculated score
        english_story = story_data.get("en", {})
        if difficulty_level and english_story:
            try:
                readability_analyzer = ReadabilityAnalyzer()
                english_pages = english_story.get("pages", [])
                full_story_text = " ".join(
                    [page.get("text", "") for page in english_pages]
                )

                if full_story_text.strip():
                    analysis_result = readability_analyzer.analyze_text(full_story_text)

                    if analysis_result and "grade_level" in analysis_result:
                        readability_score = analysis_result["grade_level"]

                        # Update the book's readability score
                        book.calculated_readability_score = readability_score
                        session.commit()

                        print(
                            f"Book {book_id}: FINAL - Target difficulty {difficulty_level}, Final readability score: {readability_score:.2f}"
                        )
                        print(f"Final analysis details: {analysis_result}")
                    else:
                        print(
                            f"Warning: Final readability analysis returned no results for book {book_id}"
                        )
            except Exception as e:
                print(
                    f"Warning: Final readability analysis failed for book {book_id}: {e}"
                )

        # Update book status and title
        book.status = "ready"
        book.title = english_story.get("title")  # Save generated title
        
        # Save model metadata if available
        model_metadata = english_story.get("model_metadata", {})
        if model_metadata:
            book.text_model = model_metadata.get("text_model")
            book.image_model = model_metadata.get("image_model")
            book.audio_model = model_metadata.get("audio_model")
            book.creation_duration = model_metadata.get("creation_duration")
            book.correct_first_try = model_metadata.get("correct_first_try")
            print(f"Book {book_id}: Saved model metadata: {model_metadata}")

        # Save pages for all languages to the DB
        for language_code, story_package in story_data.items():
            try:
                language_enum = SupportedLanguage(language_code)
            except ValueError:
                print(f"Warning: Invalid language code {language_code}, skipping")
                continue

            pages = story_package.get("pages", [])
            if pages:
                for idx, page in enumerate(pages):
                    # For translated stories, use translated_text if available
                    page_text = page.get("translated_text", page.get("text", ""))

                    page_obj = Page(
                        book_id=book_id,
                        order=idx,
                        text=page_text,
                        image_prompt=page.get("imagePrompt"),
                        audio_url=page.get("audioUrl"),
                        image_url=page.get("imageUrl"),
                        language=language_enum,
                    )
                    session.add(page_obj)

        session.commit()
        log_memory_usage("celery_worker.generate_book_task: after DB commit")

        # Explicitly delete large objects and run garbage collection
        del story_data
        del factory
        gc.collect()
        log_memory_usage("celery_worker.generate_book_task: after gc.collect()")

        return {
            "status": "ready",
            "book_id": book_id,
            "languages": list(story_data.keys())
            if "story_data" in locals()
            else ["en"],
        }

    except Exception as e:
        session.rollback()
        # Mark as failed
        book = session.execute(
            select(Book).where(Book.book_id == book_id)
        ).scalar_one_or_none()
        if book:
            book.status = "failed"
            session.commit()
        log_memory_usage("celery_worker.generate_book_task: after error")
        print(f"Error generating book {book_id}: {e}")
        return {"status": "failed", "book_id": book_id, "error": str(e)}
    finally:
        session.close()
