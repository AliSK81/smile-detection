import logging
import time


class Logger:
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')

    @staticmethod
    def log_time(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            class_name = args[0].__class__.__name__
            logging.info(f'{class_name}.{func.__name__} took {end_time - start_time:.4f} seconds')
            return result

        return wrapper
