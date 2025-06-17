import logging
import queue
import re

log_queue = queue.Queue()

class LogTextHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.timestamp_pattern = re.compile(
            r'(\[\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:,\d+)?(?:\s*[+-]\d{4})?\]|'
            r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:,\d+)?(?:\s*[+-]\d{4})?|'
            r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:,\d+)?(?:Z|[+-]\d{2}:\d{2})?)'
        )

    def emit(self, record):
        msg = record.getMessage()
        has_timestamp = bool(self.timestamp_pattern.search(msg))
        formatted_msg = msg if has_timestamp else self.format(record)
        log_queue.put(formatted_msg)