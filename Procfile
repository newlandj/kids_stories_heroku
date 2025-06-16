web: uvicorn app.main:app --host=0.0.0.0 --port=${PORT:-8000}
worker: celery -A app.celery_worker.celery_app worker --loglevel=info --max-tasks-per-child=5 --concurrency=8 --pool=threads

# Concurrency sets how many workers can run at the same time
# Max tasks per child sets how many tasks can be run before the worker is restarted. Each book creation request is a single task. 
# Pool sets the type of pool to use for the workers. Using threads vs. prefork has much lower memory usage. 
# With prefork, we were quickly hitting the memory limit. 
