from db.connection import get_db_connection
import uuid
from datetime import datetime
import os

# ---- Book/Scene DB Logic ----
def create_book(prompt: str, request_id: str) -> str:
    if os.environ.get("SKIP_DB", "false").lower() in ("true", "1", "yes"):
        return "dummy-book-id"
    status = "generating"
    created_at = datetime.utcnow()
    conn = get_db_connection()
    book_id = None
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO books (prompt, request_id, status, created_at)
                VALUES (%s, %s, %s, %s)
                RETURNING id;
                """,
                (prompt, request_id, status, created_at)
            )
            book_id = cur.fetchone()[0]
        conn.commit()
        return book_id
    finally:
        conn.close()

def fetch_book(book_id: str) -> dict:
    if os.environ.get("SKIP_DB", "false").lower() in ("true", "1", "yes"):
        return {
            "book_id": book_id,
            "prompt": "",
            "status": "complete",
            "story_text": "",
            "word_count": 0,
            "scene_count": 0,
            "scenes": []
        }
    conn = get_db_connection()
    with conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, prompt, status, story_text, word_count, scene_count FROM books WHERE id=%s", (book_id,))
            row = cur.fetchone()
            if not row:
                return None
            book = {
                "book_id": row[0],
                "prompt": row[1],
                "status": row[2],
                "story_text": row[3],
                "word_count": row[4],
                "scene_count": row[5],
            }
            if book["status"] == "complete":
                cur.execute("SELECT scene_index, image_url, audio_url FROM scenes WHERE book_id=%s ORDER BY scene_index", (book_id,))
                book["scenes"] = [
                    {"scene_index": r[0], "image_url": r[1], "audio_url": r[2]} for r in cur.fetchall()
                ]
            else:
                book["scenes"] = []
    conn.close()
    return book

def update_book_and_scenes(book_id, story_package):
    if os.environ.get("SKIP_DB", "false").lower() in ("true", "1", "yes"):
        return None
    conn = get_db_connection()
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE books SET story_text=%s, word_count=%s, scene_count=%s, status='complete' WHERE id=%s
                """,
                (
                    story_package["story_text"],
                    story_package["word_count"],
                    story_package["scene_count"],
                    book_id,
                )
            )
            scenes = zip(
                range(story_package["scene_count"]),
                story_package["visual_elements"],
                story_package["audio_narration"]
            )
            for scene_index, visual, audio in scenes:
                cur.execute(
                    """
                    INSERT INTO scenes (id, book_id, scene_index, image_url, audio_url)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        str(uuid.uuid4()),
                        book_id,
                        scene_index,
                        visual.get("image_url"),
                        audio.get("audio_url")
                    )
                )
    conn.close()

def mark_book_failed(book_id: str):
    conn = get_db_connection()
    with conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE books SET status='failed' WHERE id=%s", (book_id,))
    conn.close()
