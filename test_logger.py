import os
import sys
import time

# -----------------------------------------------------------------------------
common_path = '../common'
if common_path not in sys.path:
    sys.path.insert(0, common_path)  # insert at the beginning to prioritize

from logger import Logger

logger = Logger()

# Main entry point
if __name__ == "__main__":
    
    # We have to shut off the logger FIRST
    logger.no_log = "log" not in sys.argv
    logger.set_default_logfile("gen_log.txt")
    logger.set_rank(0)
    logger.print_and_log("Model util v0.2")