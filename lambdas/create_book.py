from shared.books import create_book
from shared.responses import api_response
import json

def lambda_handler(event, context):
    body = json.loads(event["body"])
    prompt = body.get("prompt")
    request_id = body.get("request_id")
    if not prompt:
        return api_response(400, {"error": "Missing prompt"})
    if not request_id:
        return api_response(400, {"error": "Missing request_id"})
    try:
        book_id = create_book(prompt, request_id)
        return api_response(202, {"book_id": book_id})
    except Exception as e:
        return api_response(500, {"error": str(e)})
