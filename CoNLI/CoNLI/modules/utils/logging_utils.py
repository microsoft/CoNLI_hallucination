from datetime import datetime
import logging
import os

class ContextFilter(logging.Filter):
    def __init__(self, data_id: str = None, sentence_id: str = None):
        super().__init__()
        self.data_id = data_id
        self.sentence_id = sentence_id

    def filter(self, record):
        record.data_id = self.data_id
        record.sentence_id = self.sentence_id
        return True


def get_run_id() -> str:
    # use the current time as the run id
    # yyyymmdd-hhmmss.ff
    return datetime.now().strftime("%Y%m%d-%H%M%S.%f")


def init_logging(log_level: str = "info", logfile_name: str = None):
    log_level_int = logging._nameToLevel[log_level.upper()]
    if logfile_name is None:
        today_str = datetime.now().strftime("%Y%m%d")
        filename = f'hallucination_detection-{today_str}.log'
        logfile_name = os.path.join(os.getcwd(), filename)

    run_id = get_run_id()
    logging.basicConfig(
        level=log_level_int,
        format=f'%(asctime)s %(levelname)s %(name)s run_id:{run_id} %(message)s',
        handlers=[
            logging.FileHandler(logfile_name),
            logging.StreamHandler()])

    logger = logging.getLogger()
    logger.addFilter(ContextFilter())
    print(f'logging to {logfile_name} ...')

def update_context_info(enc_id: str, sent_id: str):
    logger = logging.getLogger()
    for f in logger.filters:
        if isinstance(f, ContextFilter):
            f.data_id = enc_id
            f.sentence_id = sent_id


if __name__ == "__main__":
    print(get_run_id())

    init_logging(log_level="debug")

    logging.info("detected issue, reason: not found in src")

    update_context_info("100", "11")
    logging.info("detected issue, reason: not found in src")

    update_context_info("200", "22")
    logging.info("detected issue, reason: not found in src")

    update_context_info("300", "33")

    logging.info("hello text for 33")
