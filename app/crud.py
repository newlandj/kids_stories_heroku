import uuid
from typing import List

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.content_safety import ContentScreener
from app.models import Book, Page, SupportedLanguage


async def create_book(
    session: AsyncSession,
    prompt: str,
    request_id: str,
    title: str = None,
    difficulty_level: int = 2,
    calculated_readability_score: float = None,
):
    print("Creating book...")
    screener = ContentScreener()
    screener.validate_prompt(prompt)
    book = Book(
        book_id=uuid.uuid4(),
        request_id=request_id,
        prompt=prompt,
        status="pending",
        title=title,
        difficulty_level=difficulty_level,
        calculated_readability_score=calculated_readability_score,
    )
    session.add(book)
    try:
        await session.commit()
        print("Book created successfully")
    except IntegrityError as e:
        print("IntegrityError:", e)
        await session.rollback()
        # Duplicate request_id: fetch and return existing
        q = await session.execute(select(Book).where(Book.request_id == request_id))
        book = q.scalar_one_or_none()
        return book
    return book


async def update_book_status(session: AsyncSession, book_id: uuid.UUID, status: str):
    q = await session.execute(select(Book).where(Book.book_id == book_id))
    book = q.scalar_one_or_none()
    if not book:
        return None
    book.status = status
    await session.commit()
    return book


async def fetch_book_by_id(session: AsyncSession, book_id: uuid.UUID):
    q = await session.execute(select(Book).where(Book.book_id == book_id))
    return q.scalar_one_or_none()


async def fetch_book_by_request_id(session: AsyncSession, request_id: str):
    q = await session.execute(select(Book).where(Book.request_id == request_id))
    return q.scalar_one_or_none()


async def create_pages(
    session: AsyncSession,
    book_id: uuid.UUID,
    pages: list[dict],
    language: SupportedLanguage = SupportedLanguage.ENGLISH,
):
    """Create Page rows for a book given a list of dicts with keys: order, text, image_prompt."""
    page_objs = [
        Page(
            book_id=book_id,
            order=idx,
            language=language,
            text=page["text"],
            image_prompt=page.get("imagePrompt"),
            image_url=page.get("imageUrl"),
            audio_url=page.get("audioUrl"),
        )
        for idx, page in enumerate(pages)
    ]
    session.add_all(page_objs)
    await session.commit()
    return page_objs


async def fetch_pages_by_book_id_and_language(
    db: AsyncSession, book_id: uuid.UUID, language: SupportedLanguage
) -> List[Page]:
    """Fetch all pages for a book in a specific language, ordered by page order."""
    result = await db.execute(
        select(Page)
        .where(Page.book_id == book_id, Page.language == language)
        .order_by(Page.order)
    )
    return result.scalars().all()


async def fetch_pages_by_book_id(
    db: AsyncSession,
    book_id: uuid.UUID,
    language: SupportedLanguage = SupportedLanguage.ENGLISH,
) -> List[Page]:
    """Fetch all pages for a book in a specific language (defaults to English), ordered by page order."""
    return await fetch_pages_by_book_id_and_language(db, book_id, language)


async def fetch_all_pages_by_book_id(session: AsyncSession, book_id: uuid.UUID):
    """Fetch all pages for a book across all languages."""
    q = await session.execute(
        select(Page).where(Page.book_id == book_id).order_by(Page.order, Page.language)
    )
    return q.scalars().all()


async def fetch_available_languages_for_book(session: AsyncSession, book_id: uuid.UUID):
    """Get list of available languages for a book."""
    q = await session.execute(
        select(Page.language).where(Page.book_id == book_id).distinct()
    )
    return [lang for lang in q.scalars().all()]


async def create_translated_pages(
    session: AsyncSession,
    book_id: uuid.UUID,
    translated_pages: list[dict],
    target_language: SupportedLanguage,
):
    """Create translated page records for a book."""
    page_objs = [
        Page(
            book_id=book_id,
            order=page["order"],
            language=target_language,
            text=page["translated_text"],
            image_prompt=page.get("image_prompt"),  # Reuse original image
            image_url=page.get("image_url"),  # Reuse original image
            audio_url=page.get("audio_url"),  # Will be generated separately
        )
        for page in translated_pages
    ]
    session.add_all(page_objs)
    await session.commit()
    return page_objs


async def update_page_audio_url(
    session: AsyncSession,
    book_id: uuid.UUID,
    order: int,
    language: SupportedLanguage,
    audio_url: str,
):
    """Update the audio URL for a specific page in a specific language."""
    q = await session.execute(
        select(Page).where(
            Page.book_id == book_id, Page.order == order, Page.language == language
        )
    )
    page = q.scalar_one_or_none()
    if not page:
        return None
    page.audio_url = audio_url
    await session.commit()
    return page


async def update_book_title(session: AsyncSession, book_id: uuid.UUID, title: str):
    q = await session.execute(select(Book).where(Book.book_id == book_id))
    book = q.scalar_one_or_none()
    if not book:
        return None
    book.title = title
    await session.commit()
    return book


async def update_book_readability_score(
    session: AsyncSession, book_id: uuid.UUID, calculated_readability_score: float
):
    """Update the calculated readability score for a book."""
    q = await session.execute(select(Book).where(Book.book_id == book_id))
    book = q.scalar_one_or_none()
    if not book:
        return None
    book.calculated_readability_score = calculated_readability_score
    await session.commit()
    return book
