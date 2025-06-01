"""
Utility functions for the Audio Deepfake Detection System

This module provides helper functions for file handling, directory management,
and other common tasks.

Author: DeepfakeSoundShield Team
Date: May 2025
"""

import os
import time
import shutil
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def is_allowed_file(filename, allowed_extensions):
    """
    Check if a file has an allowed extension
    
    Parameters:
        filename (str): Name of the file to check
        allowed_extensions (set): Set of allowed file extensions
        
    Returns:
        bool: True if file extension is allowed, False otherwise
    """
    if '.' not in filename:
        return False
    
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in allowed_extensions

def create_directories(directory_list):
    """
    Create directories if they don't exist
    
    Parameters:
        directory_list (list): List of directory paths to create
    """
    for directory in directory_list:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
                logger.info(f"Created directory: {directory}")
            except Exception as e:
                logger.error(f"Error creating directory {directory}: {str(e)}")

def cleanup_old_files(directory, max_age_hours=24):
    """
    Remove files older than the specified age
    
    Parameters:
        directory (str): Directory to clean up
        max_age_hours (int): Maximum age of files in hours
    """
    if not os.path.exists(directory):
        return
    
    current_time = datetime.now()
    max_age = timedelta(hours=max_age_hours)
    
    try:
        count = 0
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            
            if os.path.isfile(file_path):
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                if current_time - file_time > max_age:
                    os.remove(file_path)
                    count += 1
        
        if count > 0:
            logger.info(f"Cleaned up {count} old files from {directory}")
    
    except Exception as e:
        logger.error(f"Error cleaning up old files in {directory}: {str(e)}")

def safe_filename(filename):
    """
    Convert a filename to a safe version
    
    Parameters:
        filename (str): Original filename
        
    Returns:
        str: Safe filename
    """
    safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
    return ''.join(c if c in safe_chars else '_' for c in filename)

def get_file_size_mb(file_path):
    """
    Get the size of a file in megabytes
    
    Parameters:
        file_path (str): Path to the file
        
    Returns:
        float: File size in MB
    """
    if not os.path.exists(file_path):
        return 0
    
    return os.path.getsize(file_path) / (1024 * 1024)