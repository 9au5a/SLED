from sklearn.metrics import f1_score
import pandas as pd
import sys
import json 
import numpy as np


md_name = sys.argv[1] #'BC_set1_0'#
gen_outputs_path = '/ukp-storage-1/kuczykowska/code/SLED/experiments/DAIC_MC/' + md_name + '/'
TEST_LABEL_PATH = '/ukp-storage-1/kuczykowska/data/daic/labels/full_test_split.csv' 

def read_labels(label_path, test = False):
    labels = pd.read_csv(label_path, sep=',')
    labels.Participant_ID = labels.Participant_ID.astype(int)
    labels.PHQ8_Binary = labels.PHQ8_Binary.astype(int)
    labels.PHQ8_Score = labels.PHQ8_Score.astype(int)
    labels.Gender = labels.Gender.astype(int)
    return labels

avrgs = ['macro', 'micro', 'weighted', 'binary']
dtype = [('key', int), ('value', 'U13')]

def score2class(score):
    if score < 5: return 0
    if score < 10: return 1
    if score < 15: return 2
    if score < 20: return 3
    return 4

def class_str2class2(text):
    text = text.lower()
    if 'no' in text: return 0
    if 'mild' in text: return 1
    if 'moderate' in text: return 2
    if 'high' in text: return 3
    if 'severe' in text: return 4
    print('ERROR in answer score2class_str: ', text)    
    
def evaluate_predictions(labels_path, ds, predictions_fn = 'generated_predictions.json'):
    print('Evaluating predictions on ' + ds + ' dataset')
    labels = read_labels(labels_path, True)
    y = [score2class(value) for value in labels.PHQ8_Score.values]
    
    path = gen_outputs_path + predictions_fn
    with open(path, 'r') as f: y_p = json.loads(f.read())
    y_p = np.array([(int(key), value) for key, value in y_p.items()],  dtype=dtype)
    y_p = np.sort(y_p, order='key')
    y_p = [class_str2class2(value) for value in y_p]
    y_p = y_p['value'].astype(int)
    for avrg in avrgs:
        print(avrg, ':', f1_score(y, y_p, average = avrg))

evaluate_predictions(TEST_LABEL_PATH, 'test')
