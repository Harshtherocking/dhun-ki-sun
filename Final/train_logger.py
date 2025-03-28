import logging
import os

class TrainLogger:
    def __init__(self, log_dir="../logs", log_file="training.log"):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        logging.basicConfig(
            filename=os.path.join(log_dir, log_file),
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.logger = logging.getLogger("TrainLogger")

    def log(self, message):
        print(message)  # Print to console
        self.logger.info(message)

# Example Usage
# logger = TrainLogger()
# logger.log("Training started...")
