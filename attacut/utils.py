# -*- coding: utf-8 -*-
import json
import os
import time
from typing import Callable, Dict, NamedTuple, Union

import yaml

class ModelParams(NamedTuple):
    name: str
    params: str

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
    return dd



