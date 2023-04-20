import os
import datetime
import json
import pickle
import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter   
import logging


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_writer(logdir):
    now = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    writer = SummaryWriter(log_dir=os.path.join(logdir, now))
    return writer

 
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger


def save_json(data, file_name):
    with open(file_name, "w", encoding="utf-8") as w:
        w.write(json.dumps(data, ensure_ascii=False, indent=4) + "\n")


def save_pickle(data, file_path):
    with open(str(file_path), "wb") as f:
        pickle.dump(data, f)


def load_pickle(input_file):
    with open(str(input_file), "rb") as f:
        data = pickle.load(f)
    return data
