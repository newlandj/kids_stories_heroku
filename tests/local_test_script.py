# tests/local_test_script.py
import os
import asyncio
import json
print('[DEBUG] Top of local_test_script.py loaded')
from dotenv import load_dotenv
from lambdas.create_book import lambda_handler as create_book_handler
from lambdas.generate_book import lambda_handler as generate_book_handler
from lambdas.get_book import lambda_handler as get_book_handler

# Load environment variables from .env file
load_dotenv()

# Configure logging (can use lambda_function's setup if desired, or keep simple)
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("local-lambda-test")

def run_lambda_simulation():
    # Ensure API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found. Ensure it is set in your .env file.")
        return

    # 1. Define the test prompt
    test_prompt = "A curious squirrel who discovers a hidden map"

    # 2. Simulate create_book Lambda (API Gateway event)
    create_event = {
        "body": json.dumps({"prompt": test_prompt, "request_id": "test-request-id"}),
        "headers": {"Content-Type": "application/json"},
        "httpMethod": "POST",
        "requestContext": {"requestId": "test-request-id", "identity": {"sourceIp": "127.0.0.1"}}
    }
    create_context = None
    create_response = create_book_handler(create_event, create_context)
    print("[create_book] Response:", create_response)
    if not (isinstance(create_response, dict) and create_response.get("statusCode", 202) in (200, 202)):
        logger.error("create_book_handler failed.")
        return
    body = create_response.get("body")
    body_json = json.loads(body) if isinstance(body, str) else body
    book_id = body_json.get("book_id")
    if not book_id:
        logger.error("No book_id returned from create_book_handler.")
        return
    print(f"Book ID: {book_id}")

    # 3. Simulate SQS trigger to generate_book Lambda
    generate_event = {
        "Records": [
            {"body": json.dumps({"book_id": book_id, "prompt": test_prompt})}
        ]
    }
    generate_context = None
    print("Invoking generate_book_handler...")
    generate_book_handler(generate_event, generate_context)
    print("[generate_book] Invoked.")

    # 4. Simulate get_book Lambda (API Gateway event)
    get_event = {
        "queryStringParameters": {"book_id": book_id},
        "httpMethod": "GET"
    }
    get_context = None
    get_response = get_book_handler(get_event, get_context)
    print("[get_book] Response:", get_response)
    # Optionally pretty-print story package
    if get_response and isinstance(get_response, dict):
        body = get_response.get("body")
        if body:
            try:
                story_json = json.loads(body) if isinstance(body, str) else body
                print(json.dumps(story_json, indent=2))
            except Exception:
                print(body)

if __name__ == "__main__":
    run_lambda_simulation()