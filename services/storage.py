import os
import boto3
from botocore.exceptions import ClientError
import uuid

def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_DEFAULT_REGION"),
    )

BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")

def upload_file_to_s3(file_data: bytes, file_type: str, extension: str) -> str:
    """
    Uploads a file to S3 and returns the public URL.
    file_data: bytes or file-like object
    file_type: 'images', 'audio', etc.
    extension: 'png', 'mp3', etc.
    """
    s3 = get_s3_client()
    unique_id = str(uuid.uuid4())
    key = f"{file_type}/{unique_id}.{extension}"
    try:
        s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=file_data)
        url = f"https://{BUCKET_NAME}.s3.amazonaws.com/{key}"
        return url
    except ClientError as e:
        raise RuntimeError(f"Failed to upload to S3: {e}")
