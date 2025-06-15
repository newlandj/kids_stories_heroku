web: uvicorn app.main:app --host=0.0.0.0 --port=${PORT:-8000}
worker: celery -A app.celery_worker.celery_app worker --loglevel=info --max-tasks-per-child=2 --concurrency=4
