# -*- coding: utf-8 -*-
import json
import os
import time
from typing import Callable, Dict, NamedTuple, Union

import yaml

from attacut import logger

log = logger.get_logger(__name__)


class ModelParams(NamedTuple):
    name: str
    params: str


class Timer:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, type, value, traceback):
        self.stop = time.time()
        diff = self.stop - self.start
        log.info("Finished block: %s with %d seconds" % (self.name, diff))

def load_training_params(path: str) -> ModelParams:
    with open("%s/params.yml" % path, "r") as f:
        params = yaml.load(f, Loader=yaml.BaseLoader)
        return ModelParams(**params)


def parse_model_params(ss: str) -> Dict[str, Union[int, float]]:
    params = dict()
    for pg in ss.split("|"):
        k, v = pg.split(":")

        if "." in v:
            params[k] = float(v)
        else:
            params[k] = int(v)
    return params


def load_dict(data_path: str) -> Dict:
    with open(data_path, "r", encoding="utf-8") as f:
        dd = json.load(f)

    log.info("loaded %d items from dict:%s" % (len(dd), data_path))

    return dd



