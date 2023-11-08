from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import sys
import json 
import numpy as np

# TO DO SET THE LABELS SET (only for classification)! see line ~37

md_name = sys.argv[1] #'BC_set1_0'#
gen_outputs_path = '/ukp-storage-1/kuczykowska/code/SLED/experiments/DAIC_R/' + md_name + '/'
TEST_LABEL_PATH = '/ukp-storage-1/kuczykowska/data/daic/labels/full_test_split.csv' 

def read_labels(label_path, test = False):
    labels = pd.read_csv(label_path, sep=',')
    labels.Participant_ID = labels.Participant_ID.astype(int)
    labels.PHQ8_Binary = labels.PHQ8_Binary.astype(int)
    labels.PHQ8_Score = labels.PHQ8_Score.astype(int)
    labels.Gender = labels.Gender.astype(int)
    return labels 

def label2points(lbl):
    lbl = lbl.lower()
    if lbl == 'never': return 0
    if lbl == 'sometimes': return 1
    if lbl == 'often': return 2
    if lbl == 'always': return 3
    print('hava a look:', lbl)
    return 4 # ??

def answers2score(id_to_class):
    dtype = [('key', int), ('value', '<U10')]
    print('id_to_class', id_to_class)
    y = np.array([(int(key), value) for key, value in id_to_class.items()],  dtype=dtype)
    print('y', y)
    y = np.sort(y, order='key')
    print('sorted y', y)
    y['value'] = [label2points(value) for value in y['value']]
    print('to value y', y)
    
    y = np.array([(key, int(value)) for key, value in y], dtype=[('key', int), ('value', int)])
    y = y.reshape(-1, 8)
    print('grouped y', y)
    y = np.array([[(id, tup[1]) for tup in row] for id, row in enumerate(y)])
    sum_y = [np.sum(d[:,1]) for d in y]
    print('summed y', sum_y)
    
    return sum_y

dtype = [('key', int), ('value', 'U13')]
def evaluate_predictions(labels_path, ds, predictions_fn = 'generated_predictions.json'):
    print('Evaluating predictions on ' + ds + ' dataset')
    labels = read_labels(labels_path, True)
    path = gen_outputs_path + predictions_fn
    with open(path, 'r') as f: y_p = json.loads(f.read())
    summed_y_p = answers2score(y_p)
        
    print('mae:', mean_absolute_error(labels.PHQ8_Score.values, summed_y_p))
    print('rmse:', mean_squared_error(labels.PHQ8_Score.values, summed_y_p, squared = False))


evaluate_predictions(TEST_LABEL_PATH, 'test')
