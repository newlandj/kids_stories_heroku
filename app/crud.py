import uuid
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from app.models import Book
from utils.content_safety import ContentScreener

async def create_book(session: AsyncSession, prompt: str, request_id: str):
    screener = ContentScreener()
    screener.validate_prompt(prompt)
    book = Book(
        book_id=uuid.uuid4(),
        request_id=request_id,
        prompt=prompt,
        status="pending",
        result=None,
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

async def update_book_status(session: AsyncSession, book_id: uuid.UUID, status: str, result: dict = None):
    q = await session.execute(select(Book).where(Book.book_id == book_id))
    book = q.scalar_one_or_none()
    if not book:
        return None
    book.status = status
    if result is not None:
        book.result = result
    await session.commit()
    return book

async def fetch_book_by_id(session: AsyncSession, book_id: uuid.UUID):
    q = await session.execute(select(Book).where(Book.book_id == book_id))
    return q.scalar_one_or_none()

async def fetch_book_by_request_id(session: AsyncSession, request_id: str):
    q = await session.execute(select(Book).where(Book.request_id == request_id))
    return q.scalar_one_or_none()
