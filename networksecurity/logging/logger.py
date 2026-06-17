import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"  ##Log time format: Month_Day_Year_Hour_Minute_Second
LOG_DIR = os.path.join(os.getcwd(), "logs")    ##Default log directory is the current working directory + "logs" folder
os.makedirs(LOG_DIR, exist_ok=True)    ##Create the log directory if it doesn't exist
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE) ## Create the full log file path by joining the log directory and log file name

logging.basicConfig(
    filename=LOG_FILE_PATH, 
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s", ##Log format: [Timestamp] LineNumber LoggerName - LogLevel - LogMessage
    level=logging.INFO,
)

logger = logging.getLogger("networksecurity") ##Create a logger instance with the name "networksecurity"
 