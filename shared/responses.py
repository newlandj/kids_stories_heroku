# Shared API response utilities for Lambda handlers
import json

def api_response(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "body": json.dumps(body),
        "headers": {"Content-Type": "application/json"}
    }
