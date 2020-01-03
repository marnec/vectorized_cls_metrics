import logging


def set_logger(level, logfile=None):
    handlers = list()
    log_formatter = logging.Formatter('%(levelname)-8s | %(message)s')

    if logfile is not None:
        file_handler = logging.FileHandler(logfile, 'a')
        file_handler.setFormatter(log_formatter)
        handlers.append(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        handlers.append(console_handler)

    logging.basicConfig(level=level, format=log_formatter, handlers=handlers)
