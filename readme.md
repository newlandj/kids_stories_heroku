More to come!

## Heroku Stuff
app url: https://kids-story-creator-5741befe1fe7.herokuapp.com/

## Testing

**To run a more local test, use the `local_test_script.py` script:**
- Look at the .env and consider which services you want to mock to save time / resources
- Use makefile command: `make test_local`

**To test locally with the local DB:**

- **Startup Postgres**
    - (Make sure your local Postgres instance is running and accessible as configured in your `.env`)

- **Startup Redis**
    - On Mac/Linux:
      ```bash
      redis-server
      ```

- **Run migrations**
    - Apply the latest database migrations:
      ```bash
      poetry run alembic upgrade head
      ```

- **Startup Celery worker**
    - In the project root:
      ```bash
      poetry run celery -A app.celery_worker.celery_app worker --loglevel=info
      ```

- **Startup FastAPI server**
    - In the project root:
      ```bash
      poetry run uvicorn app.main:app --reload
      ```

- **Use Postman to test the API**
    - You can now send requests to `http://localhost:8000` and interact with the app.
