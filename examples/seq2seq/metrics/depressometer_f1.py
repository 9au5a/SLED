from metrics.metrics import Metric 

from typing import List, Dict
import os
import importlib
from abc import ABC, abstractmethod
import inspect
import shutil

import numpy as np

from utils.decoding import decode
from datasets import load_metric as hf_load_metric
from huggingface_hub import hf_hub_download


class depressometer_f1(Metric):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.prefix = None

        self._metric = hf_load_metric('f1', keep_in_memory=True)

        self.requires_decoded = True

    def _compute_metrics(self, id_to_pred, id_to_labels) -> Dict[str, float]:
        #print(id_to_pred)
        #print(id_to_labels)
        return self._metric.compute(**self.convert_from_map_format(id_to_pred, id_to_labels), average = 'macro')

    def label2id(self, lbl):
        lbl = lbl.lower()
        if lbl == 'no': return 0
        if lbl == 'depressed': return 1
        return 2 # ??

    def convert_from_map_format(self, id_to_pred, id_to_labels):
        index_to_id = list(id_to_pred.keys())
        predictions = [id_to_pred[id_] for id_ in index_to_id]
        references = [id_to_labels[id_][0] for id_ in index_to_id]
        #print('predictions', predictions)
        #print('references', references)
        predictions = [self.label2id(p) for p in predictions]
        references = [self.label2id(p) for p in references]
        #print('predictions', predictions)
        #print('references', references)
        
        return {"predictions": predictions, "references": references}
