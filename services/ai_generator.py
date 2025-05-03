import os
import logging
import openai
import boto3
from io import BytesIO
from config import settings

# Configure logging
logger = logging.getLogger()

# Configure OpenAI client
openai.api_key = os.environ.get("OPENAI_API_KEY", settings.get("openai_api_key"))

def create_story_text(prompt):
    """
    Generate a children's story using OpenAI's GPT-4o model.
    
    Args:
        prompt (str): User prompt describing the story theme
        
    Returns:
        str: Generated story text
    """
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": "You're a talented children's author. Create a delightful, imaginative story "
                               "for children ages 4-10. Use simple language, engaging characters, and a clear "
                               "beginning, middle, and end. Include a subtle positive message or lesson."
                },
                {
                    "role": "user",
                    "content": f"Please write a children's story about: {prompt}"
                }
            ],
            temperature=0.8,
            max_tokens=1200
        )
        
        story_text = completion.choices[0].message.content
        return story_text
    
    except Exception as e:
        logger.error(f"Error generating story text: {str(e)}")
        raise Exception(f"Failed to generate story: {str(e)}")

def generate_illustrations(story_text):
    """
    Generate illustrations for different segments of the story.
    
    Args:
        story_text (str): The story content to illustrate
        
    Returns:
        list: Dictionary objects containing image URLs and related metadata
    """
    try:
        # Divide story into logical segments for illustration
        segments = _divide_into_scenes(story_text)
        illustrations = []
        
        for i, segment in enumerate(segments):
            # Create an illustration prompt based on the segment
            illustration_prompt = f"Create a colorful, child-friendly illustration for this story segment: {segment[:200]}..."
            
            response = openai.Image.create(
                model="dall-e-3",  # Using DALL-E 3 for high-quality images
                prompt=illustration_prompt,
                n=1,
                size="1024x1024"
            )
            
            # Add image data to results
            illustrations.append({
                "image_url": response.data[0].url,
                "sequence": i + 1,
                "scene_description": segment[:100] + "..."  # Abbreviated description
            })
        
        return illustrations
    
    except Exception as e:
        logger.error(f"Error generating illustrations: {str(e)}")
        raise Exception(f"Failed to create illustrations: {str(e)}")

def synthesize_audio(story_text):
    """
    Generate audio narration for the story.
    
    Args:
        story_text (str): Full story text for narration
        
    Returns:
        str: URL to the generated audio file
    """
    try:
        # Generate audio using OpenAI's TTS
        response = openai.Audio.create(
            model="tts-1",
            voice="nova",  # Child-friendly voice
            input=story_text
        )
        
        # Get audio content
        audio_data = response.content
        
        # Upload to S3 and return URL
        file_url = _upload_to_storage(audio_data, "mp3")
        return file_url
    
    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}")
        raise Exception(f"Failed to create audio narration: {str(e)}")

def _divide_into_scenes(text):
    """
    Split a story into logical scenes for illustration.
    
    Args:
        text (str): Full story text
        
    Returns:
        list: Story segments for illustration
    """
    # Split by paragraphs
    paragraphs = [p for p in text.split('\n\n') if p.strip()]
    
    # If very few paragraphs, use them directly
    if len(paragraphs) <= 4:
        return paragraphs
    
    # Otherwise, create 4 logical segments
    segment_size = max(1, len(paragraphs) // 4)
    segments = []
    
    for i in range(0, len(paragraphs), segment_size):
        segment = ' '.join(paragraphs[i:i+segment_size])
        segments.append(segment)
        if len(segments) >= 4:  # Cap at 4 illustrations
            break
            
    return segments

def _upload_to_storage(file_data, extension):
    """
    Upload file to S3 bucket and return the URL.
    
    Note: This is a placeholder implementation. For simplicity, it returns a mock URL.
    In production, this would use AWS S3 to store the audio file.
    """
    
    # For development/testing, return a placeholder URL
    # In production, implement actual S3 upload:
    
    """
    # Actual S3 implementation would look like:
    try:
        s3_client = boto3.client('s3')
        bucket_name = settings.get("s3_bucket_name")
        file_key = f"story_narrations/{uuid.uuid4()}.{extension}"
        
        s3_client.upload_fileobj(
            BytesIO(file_data),
            bucket_name,
            file_key,
            ExtraArgs={'ContentType': f'audio/{extension}'}
        )
        
        return f"https://{bucket_name}.s3.amazonaws.com/{file_key}"
    except Exception as e:
        logger.error(f"Error uploading to S3: {str(e)}")
        raise
    """
    
    # Placeholder implementation
    return f"https://example-bucket.s3.amazonaws.com/stories/narration-{hash(str(file_data))[:8]}.{extension}"
