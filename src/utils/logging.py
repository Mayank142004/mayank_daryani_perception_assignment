import logging
import sys
import time

def setup_logger():
    logger = logging.getLogger('lane_topology')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        ))
        logger.addHandler(handler)
    return logger

logger = setup_logger()

def log_stage(stage_name: str):
    logger.info(f"{'='*10} STAGE: {stage_name.upper()} {'='*10}")

class Timer:
    """Context manager for timing code blocks."""
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.start
        logger.info(f"{self.name} took {elapsed:.3f}s")
