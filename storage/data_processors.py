import json
import base64
import logging
from typing import Dict, Any, Union

logger = logging.getLogger("kids-story-lambda")

class EventParser:
    """Utility for parsing Lambda event data"""
    
    @staticmethod
    def extract_payload(event: Dict[str, Any]) -> Dict[str, Any]:
        """Extract request data from various Lambda event sources"""
        # Check for API Gateway event structure
        if "body" in event:
            body = event["body"]
            
            # Handle base64 encoded body
            if event.get("isBase64Encoded", False):
                try:
                    body = base64.b64decode(body).decode("utf-8")
                except Exception as e:
                    logger.error(f"Failed to decode base64 body: {str(e)}")
                    raise ValueError("Invalid base64 encoding in request body")
            
            # Parse JSON body
            if isinstance(body, str):
                try:
                    return json.loads(body)
                except json.JSONDecodeError:
                    logger.error("Failed to parse JSON body")
                    raise ValueError("Request body is not valid JSON")
            else:
                return body
                
        # Check for SNS message
        elif "Records" in event and len(event["Records"]) > 0:
            if "Sns" in event["Records"][0]:
                try:
                    message = event["Records"][0]["Sns"]["Message"]
                    return json.loads(message)
                except (json.JSONDecodeError, KeyError):
                    logger.error("Failed to parse SNS message")
                    raise ValueError("Invalid SNS message format")
        
        # Direct Lambda invocation
        return event


class ResponseBuilder:
    """Utility for building Lambda responses"""
    
    @staticmethod
    def build(status_code: int, body: Dict[str, Any]) -> Dict[str, Any]:
        """Build a response compatible with API Gateway"""
        return {
            "statusCode": status_code,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type,X-Api-Key"
            },
            "body": json.dumps(body),
            "isBase64Encoded": False
        }


class StorageManager:
    """Handles interactions with S3 storage"""
    
    @staticmethod
    def generate_unique_path(file_type: str, extension: str) -> str:
        """Generate a unique path for storing content"""
        import uuid
        import datetime
        
        today = datetime.datetime.now().strftime("%Y/%m/%d")
        unique_id = str(uuid.uuid4())
        
        return f"stories/{file_type}/{today}/{unique_id}.{extension}"
