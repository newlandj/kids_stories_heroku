from dotenv import load_dotenv
load_dotenv()

import os
import asyncio
import gc
from celery import Celery
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from app.models import Book, Page, SupportedLanguage
from app.crud import create_pages, update_book_readability_score
from app.db import Base, DATABASE_URL
from app.narrative_engine import FableFactory
from app.utils import log_memory_usage
from app.readability_analyzer import ReadabilityAnalyzer

redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

import ssl

redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
redis_use_ssl = redis_url.startswith('rediss://')

ssl_options = None
if redis_use_ssl:
    ssl_options = {
        'ssl_cert_reqs': ssl.CERT_NONE,
        'ssl_check_hostname': False,
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
    broker_connection_max_retries=3
)

# Set up sync SQLAlchemy for Celery
db_url_sync = DATABASE_URL.replace("+asyncpg", "") if "+asyncpg" in DATABASE_URL else DATABASE_URL
engine = create_engine(db_url_sync)
Session = sessionmaker(bind=engine)

@celery_app.task(bind=True)
def generate_book_task(self, book_id, prompt, target_languages=None, difficulty_level=None):
    """
    Generate a book with optional multi-language support.
    
    Args:
        book_id: UUID of the book to generate
        prompt: Story generation prompt
        target_languages: Optional list of language codes to translate to
        difficulty_level: Optional difficulty level for readability analysis
    """
    log_memory_usage("celery_worker.generate_book_task: start")
    session = Session()
    try:
        # Mark as in_progress
        book = session.execute(select(Book).where(Book.book_id == book_id)).scalar_one_or_none()
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
        
        # Generate story package with readability checking and potential retry
        factory = FableFactory()
        
        # Step 1: Generate initial English story
        english_story = asyncio.run(factory.generate_story_package(prompt, difficulty_level))
        log_memory_usage("celery_worker.generate_book_task: after initial story generation")
        
        # Step 2: Check readability and potentially retry
        if difficulty_level is not None:
            try:
                readability_analyzer = ReadabilityAnalyzer()
                # Get full story text from all English pages
                english_pages = english_story.get("pages", [])
                full_story_text = " ".join([page.get("text", "") for page in english_pages])
                
                if full_story_text.strip():
                    analysis_result = readability_analyzer.analyze_text(full_story_text)
                    
                    if analysis_result and 'grade_level' in analysis_result:
                        calculated_score = analysis_result['grade_level']
                        score_difference = abs(calculated_score - difficulty_level)
                        
                        print(f"Book {book_id}: Target={difficulty_level}, Calculated={calculated_score:.2f}, Difference={score_difference:.2f}")
                        
                        # Retry if more than 1 grade level off
                        if score_difference > 1.0:
                            print(f"Book {book_id}: Retrying due to readability mismatch (>{calculated_score:.2f} vs target {difficulty_level})")
                            
                            # Generate retry with feedback
                            retry_story = asyncio.run(
                                factory.generate_story_with_readability_feedback(
                                    prompt, difficulty_level, calculated_score, english_story
                                )
                            )
                            log_memory_usage("celery_worker.generate_book_task: after readability retry")
                            
                            # Analyze the retry
                            retry_pages = retry_story.get("pages", [])
                            retry_text = " ".join([page.get("text", "") for page in retry_pages])
                            
                            if retry_text.strip():
                                retry_analysis = readability_analyzer.analyze_text(retry_text)
                                
                                if retry_analysis and 'grade_level' in retry_analysis:
                                    retry_score = retry_analysis['grade_level']
                                    retry_difference = abs(retry_score - difficulty_level)
                                    
                                    print(f"Book {book_id}: Retry score={retry_score:.2f}, difference={retry_difference:.2f}")
                                    
                                    # Keep whichever story is closer to target
                                    if retry_difference < score_difference:
                                        print(f"Book {book_id}: Using retry version (closer to target)")
                                        english_story = retry_story
                                        calculated_score = retry_score
                                    else:
                                        print(f"Book {book_id}: Keeping original version (closer to target)")
                                else:
                                    print(f"Book {book_id}: Could not analyze retry, keeping original")
                            else:
                                print(f"Book {book_id}: Retry produced no text, keeping original")
                        else:
                            print(f"Book {book_id}: Readability within acceptable range")
                            
                    else:
                        print(f"Warning: Readability analysis returned no results for book {book_id}")
                else:
                    print(f"Warning: No story text found for readability analysis for book {book_id}")
            except Exception as e:
                print(f"Warning: Readability analysis/retry failed for book {book_id}: {e}")
                # Don't fail the entire task if readability analysis fails
        
        # Step 3: Generate translations if requested
        if target_language_enums:
            # Generate with translations using the final English story
            story_data = asyncio.run(factory.generate_story_with_translations(prompt, [SupportedLanguage.ENGLISH] + target_language_enums))
            # Replace the English story with our potentially retried version
            story_data["en"] = english_story
            log_memory_usage("celery_worker.generate_book_task: after multi-language story generation")
        else:
            # English only
            story_data = {"en": english_story}
            log_memory_usage("celery_worker.generate_book_task: after English story generation")
        
        # Update the book's readability score with final calculated score
        english_story = story_data.get("en", {})
        if difficulty_level and english_story:
            try:
                readability_analyzer = ReadabilityAnalyzer()
                english_pages = english_story.get("pages", [])
                full_story_text = " ".join([page.get("text", "") for page in english_pages])
                
                if full_story_text.strip():
                    analysis_result = readability_analyzer.analyze_text(full_story_text)
                    
                    if analysis_result and 'grade_level' in analysis_result:
                        readability_score = analysis_result['grade_level']
                        
                        # Update the book's readability score
                        book.calculated_readability_score = readability_score
                        session.commit()
                        
                        print(f"Book {book_id}: FINAL - Target difficulty {difficulty_level}, Final readability score: {readability_score:.2f}")
                        print(f"Final analysis details: {analysis_result}")
                    else:
                        print(f"Warning: Final readability analysis returned no results for book {book_id}")
            except Exception as e:
                print(f"Warning: Final readability analysis failed for book {book_id}: {e}")
        
        # Update book with result
        book.status = "ready"
        book.title = english_story.get("title")  # Save generated title
        
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
                        language=language_enum
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
            "languages": list(story_data.keys()) if 'story_data' in locals() else ["en"]
        }
        
    except Exception as e:
        session.rollback()
        # Mark as failed
        book = session.execute(select(Book).where(Book.book_id == book_id)).scalar_one_or_none()
        if book:
            book.status = "failed"
            session.commit()
        log_memory_usage("celery_worker.generate_book_task: after error")
        print(f"Error generating book {book_id}: {e}")
        return {"status": "failed", "book_id": book_id, "error": str(e)}
    finally:
        session.close()
