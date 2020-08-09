import logging

logger_setup = True

root_logger = logging.getLogger()
root_logger.setLevel("DEBUG")

streamHandler = logging.StreamHandler()
streamHandler.setFormatter(logging.Formatter(
    "%(asctime)s %(name)-25s %(threadName)s %(levelname)-6s %(message)s"))

root_logger.addHandler(streamHandler)