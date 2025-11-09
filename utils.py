# LumenTact/utils.py
import logging
import sys

def setup_logger(name, level=logging.INFO):
    """
    Sets up a simple logger that prints to stdout.
    """
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent duplicate logs if already configured
    if not logger.hasHandlers():
        logger.addHandler(handler)
        
    logger.propagate = False
    
    return logger