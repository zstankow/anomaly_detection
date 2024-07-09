import logging
import logging.handlers
import queue

log_queue = queue.Queue()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.handlers.RotatingFileHandler('sensor.log', maxBytes=10*1024*1024, backupCount=5)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

queue_handler = logging.handlers.QueueHandler(log_queue)
logger.addHandler(queue_handler)

listener = logging.handlers.QueueListener(log_queue, file_handler)
listener.start()
