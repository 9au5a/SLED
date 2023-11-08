from metrics.metrics import Metric 

from typing import List, Dict
import os
import importlib
from abc import ABC, abstractmethod
import inspect
import shutil

import numpy as np
import evaluate 
from utils.decoding import decode
from datasets import load_metric as hf_load_metric
from huggingface_hub import hf_hub_download

class RMSE(Metric):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.prefix = None

        self._metric = hf_load_metric('mse', keep_in_memory=True)
        #self._metric = evaluate.load("mae")
        self.requires_decoded = True
    
    def _compute_metrics(self, id_to_pred, id_to_labels) -> Dict[str, float]:
        #print(id_to_pred)
        #print(id_to_labels)
        predictions = self.answers2score(id_to_pred, in_array = True)
        references = self.answers2score(id_to_labels)
        
        return self.calculate_metrics(predictions, references)

    def label2points(self, lbl):
        lbl = lbl.lower()
        if lbl == 'never': return 0
        if lbl == 'sometimes': return 1
        if lbl == 'often': return 2
        if lbl == 'always': return 3
        print('hava a look:', lbl)
        return 4 # ??

    def answers2score(self, id_to_class, in_array = False):
        dtype = [('key', int), ('value', '<U10')]
        #print('id_to_label', id_to_class)
        if in_array:
            y = np.array([(int(key), value) for key, value in id_to_class.items()],  dtype=dtype)
        else:
            y = np.array([(int(key), value[0]) for key, value in id_to_class.items()],  dtype=dtype)
        y = np.sort(y, order='key')
        y['value'] = [self.label2points(value) for value in y['value']]
        
        y = np.array([(key, int(value)) for key, value in y], dtype=[('key', int), ('value', int)])
        y = y.reshape(-1, 8)
        #print('grouped y', y)
        y = np.array([[(id, tup[1]) for tup in row] for id, row in enumerate(y)])
        sum_y = [np.sum(d[:,1]) for d in y]
        print('summed y', sum_y)
        
        return sum_y
    
    def calculate_metrics(self, predictions, references):
        print('RMSE')
        return self._metric.compute(predictions=predictions, references=references, squared = False)
