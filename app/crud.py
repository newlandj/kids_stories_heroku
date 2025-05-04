import uuid
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from app.models import Book, Page
from utils.content_safety import ContentScreener

async def create_book(session: AsyncSession, prompt: str, request_id: str):
    screener = ContentScreener()
    screener.validate_prompt(prompt)
    book = Book(
        book_id=uuid.uuid4(),
        request_id=request_id,
        prompt=prompt,
        status="pending"
    )
    session.add(book)
    try:
        await session.commit()
    except IntegrityError:
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

async def create_pages(session: AsyncSession, book_id: uuid.UUID, pages: list[dict]):
    """Create Page rows for a book given a list of dicts with keys: order, text, image_prompt."""
    page_objs = [
        Page(
            book_id=book_id,
            order=idx,
            text=page["text"],
            image_prompt=page.get("imagePrompt"),
            image_url=page.get("imageUrl"),
            audio_url=page.get("audioUrl")
        )
        for idx, page in enumerate(pages)
    ]
    session.add_all(page_objs)
    await session.commit()
    return page_objs

async def fetch_pages_by_book_id(session: AsyncSession, book_id: uuid.UUID):
    q = await session.execute(select(Page).where(Page.book_id == book_id).order_by(Page.order))
    return q.scalars().all()
