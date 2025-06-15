import logging
import os


def log_memory_usage(context: str = ""):
    """
    Logs current and peak memory usage (RSS) for the current process.
    Args:
        context (str): Optional description of where this log is called.
    """
    try:
        import psutil

        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / (1024 * 1024)  # in MB
        logging.info(f"[MEMORY] {context} | RSS: {mem:.2f} MB")
    except ImportError:
        logging.warning("psutil not installed; memory usage logging skipped.")
    except Exception as e:
        logging.error(f"Error logging memory usage: {e}")
