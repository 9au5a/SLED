from sklearn.metrics import f1_score
import pandas as pd
import sys
import json 
import numpy as np

# TO DO SET THE LABELS SET! see line ~37

md_name = sys.argv[1] #'BC_set1_0'#
gen_outputs_path = '/ukp-storage-1/kuczykowska/code/SLED/experiments/DAIC_BC/' + md_name + '/'

def read_labels(label_path, test = False):
    labels = pd.read_csv(label_path, sep=',')
    labels.Participant_ID = labels.Participant_ID.astype(int)
    labels.PHQ8_Binary = labels.PHQ8_Binary.astype(int)
    labels.PHQ8_Score = labels.PHQ8_Score.astype(int)
    labels.Gender = labels.Gender.astype(int)
    
    return labels

TEST_LABEL_PATH = '/ukp-storage-1/kuczykowska/data/daic/labels/full_test_split.csv' 

avrgs = ['macro', 'micro', 'weighted', 'binary']
dtype = [('key', int), ('value', 'U13')]
def evaluate_predictions(labels_path, ds, predictions_fn = 'generated_predictions.json'):
    print('Evaluating predictions on ' + ds + ' dataset')
    labels = read_labels(labels_path, True)
    path = gen_outputs_path + predictions_fn
    with open(path, 'r') as f: y_p = json.loads(f.read())
    ny_p = np.array([(int(key), value) for key, value in y_p.items()],  dtype=dtype)
    ny_p = np.sort(ny_p, order='key')
    #ny_p['value'] = [1 if value.lower() == 'depressed' else 0 for value in ny_p['value']]
    ny_p['value'] = [1 if value.lower() == 'yes' else 0 for value in ny_p['value']]
    y_p = ny_p['value'].astype(int)
        
    for avrg in avrgs:
        print(avrg, ':', f1_score(labels.PHQ8_Binary.values, y_p, average = avrg))


evaluate_predictions(TEST_LABEL_PATH, 'test')
