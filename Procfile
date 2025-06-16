web: uvicorn app.main:app --host=0.0.0.0 --port=${PORT:-8000}
# worker: celery -A app.celery_worker.celery_app worker --loglevel=info --max-tasks-per-child=1 --concurrency=10
worker: celery -A app.celery_worker.celery_app worker --loglevel=info --max-tasks-per-child=5 --concurrency=8 --pool=threads
