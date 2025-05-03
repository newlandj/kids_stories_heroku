# Migration Plan: AWS Lambda to FastAPI on Heroku

## 1. Project Setup & Initial FastAPI App
- Create a new FastAPI app structure suitable for Heroku deployment.
- Set up requirements.txt for dependencies (FastAPI, Uvicorn, etc.).
- Add Procfile for Heroku.
- Configure environment variables and settings for Heroku compatibility.

## 2. Endpoint Mapping & Refactoring
- Analyze existing Lambda functions:
  - Book creation trigger (receives request, triggers book generation)
  - Book generation (generate_book.py)
  - Book retrieval/status (get_book.py)
- Map each Lambda to a FastAPI route:
  - POST `/books/` to initiate book creation.
  - GET `/books/{book_id}` to poll for status and retrieve completed books.

## 3. Background Task Handling
- Since book generation takes ~45 seconds, use FastAPI’s background tasks or a task queue (e.g., Celery with Redis) for asynchronous processing.
  - For simplicity and Heroku compatibility, use FastAPI’s built-in BackgroundTasks if scaling is not a concern.
  - Store book status and results in a persistent store (e.g., Postgres, Redis, or file storage).
- Update endpoints to:
  - Return a job/book ID immediately upon creation.
  - Allow polling for status and retrieval of the finished book.

## 4. Data Storage Refactor
- Replace S3/DynamoDB usage with Heroku-friendly alternatives:
  - Use Postgres (Heroku Postgres add-on) for metadata and status.
  - Use filesystem or Postgres for storing generated books, or a third-party storage service if needed.

## 5. Code Migration
- Refactor Lambda code into FastAPI route handlers and background task functions.
- Remove AWS-specific code (event/context, boto3, etc.).
- Adapt SQS/S3 logic to new storage and background task mechanism.

## 6. Testing & Validation
- Add unit and integration tests for new endpoints and background processing.
- Test locally and on Heroku.

## 7. Deployment
- Push code to Heroku Git or connect to GitHub.
- Set up environment variables and add-ons (e.g., Postgres).
- Deploy and verify app functionality.

## 8. Documentation
- Update README with new API usage and deployment instructions.

---

### Notes on Background Processing

- For Heroku, FastAPI’s BackgroundTasks is simplest but will not persist if the dyno restarts. For more robust, production-grade processing, consider using Celery with Redis or a managed task queue.
- For book status/results, use a database (e.g., Postgres) to track each book’s progress and result.
