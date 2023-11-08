from sled import SledTokenizer, SledModel
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
import torch
from run import preprocess_function
from copy import deepcopy
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error
import numpy as np

PREFIX_DOC_SEP = '\n\n'
max_source_length = 16384

md_name = 'R2'
gen_outputs_path = '/ukp-storage-1/kuczykowska/code/SLED/tmp/generated_outputs/'

path = '/ukp-storage-1/kuczykowska/code/SLED/tmp/' +  md_name
model = AutoModelForSeq2SeqLM.from_pretrained(path)

tokenizer_path = '/ukp-storage-1/kuczykowska/code/SLED/tmp/'+ md_name +'/config.json'
tokenizer = SledTokenizer.from_pretrained(tokenizer_path)

def preprocess_function_kwargs_fn():
        return {
            "tokenizer": deepcopy(tokenizer),
            "prefix": '',
            "input_column": 'input',
            "input_prefix_column": 'input_prefix',
            "output_column": 'output',
            "max_source_length": max_source_length,
            "max_prefix_length": 128,
            "max_target_length": 8,
            "prefix_sep": PREFIX_DOC_SEP,
            "padding": False,
            "ignore_pad_token_for_loss": True,
            "assign_zero_to_too_long_val_examples": False,
            "trim_very_long_strings": False,
            "pad_prefix": True
        }

preprocess_function_kwargs = preprocess_function_kwargs_fn()

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

def predict(test_dataset, namefn):        
    predict_dataset = test_dataset.map(
                    preprocess_function,
                    fn_kwargs=preprocess_function_kwargs,
                    batched=True,
                    num_proc=None,
                    remove_columns=set(test_dataset.column_names).difference(set(['id'])),
                    load_from_cache_file=False,
                    desc="Running tokenizer on prediction dataset",
                )

    predict_dataset = predict_dataset.with_format('torch', columns = set(predict_dataset.column_names).difference(set(['id'])))
    print(predict_dataset)
    outputs = dict()
    for r_index, input_ids, prfl, attm in zip(predict_dataset["id"]
                         , predict_dataset["input_ids"]
                         #, predict_dataset["input_prefix"]
                         , predict_dataset["prefix_length"]
                         , predict_dataset["attention_mask"]):
        with torch.no_grad():
            generated_output = model.generate(input_ids=input_ids.reshape(torch.Size([1, input_ids.shape[0]]))
                                    , attention_mask = attm.reshape(torch.Size([1, attm.shape[0]]))
                                    #, input_prefix = input_prefix.reshape(torch.Size([1, input_prefix.shape[0]]))
                                    , prefix_length = torch.LongTensor([[prfl]])
                                    , return_dict=None)
            decoded = tokenizer.decode(generated_output[0]).lower()
            output = ''
            if 'never' in decoded: output = 'Never' 
            elif 'sometimes' in decoded: output = 'Sometimes' 
            elif 'often' in decoded: output = 'Often' 
            elif 'always' in decoded: output = 'Always' 
            outputs[r_index] =  output
        
    summed_y = answers2score(outputs)
    print(outputs)
    print(summed_y)
    with open(gen_outputs_path + namefn + '_c.csv', 'w') as f: 
        f.write('\n'.join(summed_y))
    #with open(gen_outputs_path + namefn + '_r.csv', 'w') as f: 
    #    f.write('\n'.join(outputs))



OUT_PATH  = '/ukp-storage-1/kuczykowska/data/daic/created_datasets/regression_dataset'
test_dataset = load_dataset(OUT_PATH, split = 'test', cache_dir='./', )
predict(test_dataset, md_name + '_test')
test_dataset = load_dataset(OUT_PATH, split = 'validation', cache_dir='./', )
predict(test_dataset, md_name + '_dev')


import pandas as pd

def read_labels(label_path, test = False):
    labels = pd.read_csv(label_path, sep=',')
    labels.Participant_ID = labels.Participant_ID.astype(int)
    labels.PHQ8_Binary = labels.PHQ8_Binary.astype(int)
    labels.PHQ8_Score = labels.PHQ8_Score.astype(int)
    labels.Gender = labels.Gender.astype(int)
    if not test:
        for f in phq8_features:
            labels[f] = labels[f].astype(int)
    return labels 


TEST_LABEL_PATH = '/ukp-storage-1/kuczykowska/data/daic/labels/full_test_split.csv' 
DEV_LABEL_PATH = '/ukp-storage-1/kuczykowska/data/daic/labels/dev_split_Depression_AVEC2017.csv' 

def evaluate_predictions(path, ds):
    print('Evaluating predictions on ' + ds + ' dataset')
    labels = read_labels(path, True)
    y_p = pd.read_csv(gen_outputs_path + md_name + '_' + ds + '_r.csv', header = None)
    print('mae:', mean_absolute_error(labels.PHQ8_Score.values, y_p))
    print('rmse:', mean_squared_error(labels.PHQ8_Score.values, y_p, squared = False))

evaluate_predictions(TEST_LABEL_PATH, 'test')
evaluate_predictions(DEV_LABEL_PATH, 'dev')
