import logging


def init_stream_logger(name: str, loglevel: str = "INFO"):
    """Initialise a logger to stdout
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logger = logging.getLogger(name)

    # To override the default severity of logging
    logger.setLevel(loglevel)

    # Use StreamHandler() to log to the console
    handler = logging.StreamHandler()
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger
