from shared.books import update_book_and_scenes, mark_book_failed
from storytelling.narrative_engine import generate_story_package
import json

def lambda_handler(event, context):
    for record in event["Records"]:
        body = json.loads(record["body"])
        book_id = body["book_id"]
        prompt = body["prompt"]
        try:
            story_package = generate_story_package(prompt)
            update_book_and_scenes(book_id, story_package)
        except Exception as e:
            mark_book_failed(book_id)
