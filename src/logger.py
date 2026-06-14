# src/logger.py
import logging
import os

def get_logger(name):
    """Sets up a logger that outputs to both the console and a log file."""
    os.makedirs("outputs/logs", exist_ok=True)
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        
        # Output to console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Output to file
        file_handler = logging.FileHandler("outputs/logs/pipeline.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger