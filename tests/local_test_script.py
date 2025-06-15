"""
Test script for FableFactory story pipeline.

Modes:
- LIVE:         All real services (default)
- MOCK_SERVICES: Patch OpenAI, S3, etc. (MOCK_SERVICES=true)
- HYBRID:       Only patch S3 upload (MOCK_S3=true), OpenAI is live

Usage:
  poetry run python -m tests.local_test_script           # Live
  MOCK_SERVICES=true poetry run python -m tests.local_test_script  # Full mock
  MOCK_S3=true poetry run python -m tests.local_test_script        # Hybrid (real OpenAI, dummy S3)
"""

import asyncio
import json
import logging
import os
from unittest import mock

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# If MOCK_SERVICES is set, automatically enable USE_DUMMY_AI
if os.environ.get("MOCK_SERVICES", "false").lower() in ("true", "1", "yes"):
    os.environ["USE_DUMMY_AI"] = "true"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("story-pipeline-test")

# Import the story pipeline
from app.narrative_engine import FableFactory


def mock_openai_generate_story(*args, **kwargs):
    return "Once upon a time, there was a mock squirrel who found a mock map."


def mock_identify_key_pages(story_text):
    return ["Mock Page 1", "Mock Page 2", "Mock Page 3"]


def mock_generate_single_illustration(page_prompt, idx, art_direction):
    return f"https://mock-s3-bucket.amazonaws.com/images/mock-illustration-{idx}.png"


def mock_record_narration(pages):
    return [
        {
            "audio_url": f"https://mock-s3-bucket.amazonaws.com/audio/mock-narration-{i}.mp3",
            "page_index": i,
        }
        for i, _ in enumerate(pages)
    ]


def mock_upload_file_to_s3(file_bytes, file_type, extension):
    return f"https://mock-s3-bucket.amazonaws.com/{file_type}/mock-file.{extension}"


async def run_story_pipeline_test():
    test_prompt = "A curious squirrel who discovers a hidden map"
    mock_mode = os.environ.get("MOCK_SERVICES", "false").lower() in ("true", "1", "yes")
    hybrid_mode = os.environ.get("MOCK_S3", "false").lower() in ("true", "1", "yes")

    if mock_mode:
        logger.info("Running in MOCK mode!")
        with (
            mock.patch.object(
                FableFactory, "weave_narrative", side_effect=mock_openai_generate_story
            ),
            mock.patch.object(
                FableFactory, "_identify_key_pages", side_effect=mock_identify_key_pages
            ),
            mock.patch.object(
                FableFactory,
                "_generate_single_illustration",
                side_effect=mock_generate_single_illustration,
            ),
            mock.patch.object(
                FableFactory, "record_narration", side_effect=mock_record_narration
            ),
            mock.patch(
                "app.storage.upload_file_to_s3", side_effect=mock_upload_file_to_s3
            ),
        ):
            factory = FableFactory()
            result = await factory.generate_story_package(test_prompt)
    elif hybrid_mode:
        logger.info("Running in HYBRID mode (real OpenAI, dummy S3 upload)!")
        with mock.patch(
            "app.storage.upload_file_to_s3", side_effect=mock_upload_file_to_s3
        ):
            factory = FableFactory()
            result = await factory.generate_story_package(test_prompt)
    else:
        logger.info("Running in LIVE mode!")
        factory = FableFactory()
        result = await factory.generate_story_package(test_prompt)

    print("\n===== STORY PACKAGE RESULT =====\n")
    print(json.dumps(result, indent=2))
    # Assertions for new structured format
    assert "title" in result and isinstance(result["title"], str)
    assert "characters" in result and isinstance(result["characters"], list)
    assert (
        "pages" in result
        and isinstance(result["pages"], list)
        and len(result["pages"]) > 0
    )
    assert "visual_elements" in result and isinstance(result["visual_elements"], list)
    assert "audio_narration" in result and isinstance(result["audio_narration"], list)
    for page in result["pages"]:
        assert "text" in page and isinstance(page["text"], str)
        assert "imagePrompt" in page and isinstance(page["imagePrompt"], str)
    assert all("http" in url for url in result["visual_elements"]), (
        "Image URLs missing or invalid"
    )
    assert all("audio_url" in a for a in result["audio_narration"]), (
        "Audio URLs missing"
    )
    print("\nTest completed successfully!\n")


if __name__ == "__main__":
    asyncio.run(run_story_pipeline_test())
