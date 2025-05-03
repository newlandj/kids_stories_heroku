from shared.books import fetch_book
from shared.responses import api_response

def lambda_handler(event, context):
    params = event.get("pathParameters") or event.get("queryStringParameters") or {}
    book_id = params.get("book_id")
    if not book_id:
        return api_response(400, {"error": "Missing book_id parameter"})
    try:
        book = fetch_book(book_id)
        if not book:
            return api_response(404, {"error": "Book not found"})
        return api_response(200, book)
    except Exception as e:
        return api_response(500, {"error": str(e)})
