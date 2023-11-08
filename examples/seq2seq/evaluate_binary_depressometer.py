from sklearn.metrics import f1_score
import pandas as pd
import sys
import json 
import numpy as np
from datasets import load_dataset

md_name = sys.argv[1] 
gen_outputs_path = '/ukp-storage-1/kuczykowska/code/SLED/experiments/depressometer_BC/' + md_name + '/'
OUT_PATH  = '/ukp-storage-1/kuczykowska/data/depressometer/created_datasets/classification_binary_dataset'   

avrgs = ['macro', 'micro', 'weighted', 'binary']
dtype = [('key', int), ('value', 'U13')]

def evaluate_predictions(labels, ds, predictions_fn = 'generated_predictions.json'):
    print('Evaluating predictions on ' + ds + ' dataset')
    y = [1 if value.lower() == 'depressed' else 0 for value in labels]

    path = gen_outputs_path + predictions_fn
    with open(path, 'r') as f: y_p = json.loads(f.read())
    y_p = np.array([(int(key), value) for key, value in y_p.items()],  dtype=dtype)
    y_p = np.sort(y_p, order='key')
    y_p['value'] = [1 if value.lower() == 'depressed' else 0 for value in y_p['value']]
    y_p = y_p['value'].astype(int)
        
    for avrg in avrgs:
        print(avrg, ':', f1_score(y, y_p, average = avrg))


test_dataset = load_dataset(OUT_PATH, split = 'test', cache_dir='./', )
test_labels = test_dataset['output']
evaluate_predictions(test_labels, 'test')
