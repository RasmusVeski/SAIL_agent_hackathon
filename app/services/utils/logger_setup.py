import logging
import sys
import os
import datetime

def setup_logging(log_dir="logs", log_file="training.log"):
    """
    Configures the root logger.

    Moves the previous log file to an 'archive' sub-directory 
    within the 'log_dir' with a timestamp.

    Parameters:
    - log_dir (str): Directory to store log files.
    - log_file (str): Name of the current log file.
    """
    
    # Define paths
    log_path = os.path.join(log_dir, log_file)
    archive_dir = os.path.join(log_dir, "archive") # <-- Old logs go here

    # Create directories if they don't exist
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(archive_dir, exist_ok=True)

    # --- Archive previous log if it exists ---
    if os.path.exists(log_path):
        try:
            # Create a timestamped name for the old log
            now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            base_name, ext = os.path.splitext(log_file)
            archive_name = f"{base_name}_{now}{ext}"
            
            # Full path for the archived log
            archive_file_path = os.path.join(archive_dir, archive_name)

            # Move the old log file
            os.rename(log_path, archive_file_path)
        except Exception as e:
            print(f"Warning: Could not archive old log file: {e}")
            # Continue anyway, will overwrite

    # set up logging for the new run
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) # Set level on the logger itself
    

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

     ## 1. Check for a FileHandler
    # We check the *specific file path* to avoid issues if another
    # process (like Agent_B) already set up a *different* file handler.
    # In our case, they are separate processes, so this check is simpler.
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        # --- File Handler (for current log) ---
        fh = logging.FileHandler(log_path, mode='w')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logging.info("File logging configured. Outputting to %s", log_path)
        if 'archive_file_path' in locals():
            logging.info("Archived previous log to %s", archive_file_path)
    else:
        logging.info("FileHandler already configured.")

    # 2. Check for a StreamHandler (Console)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        # --- Console (Stream) Handler ---
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        logging.info("Console logging configured.")
    else:
        logging.info("StreamHandler already configured.")