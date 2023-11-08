from sled import SledTokenizer, SledModel
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
import torch
from run import preprocess_function
from copy import deepcopy
from sklearn.metrics import f1_score


PREFIX_DOC_SEP = '\n\n'
max_source_length = 16384


md_name = 'T12'
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
            "pad_prefix": False
        }

preprocess_function_kwargs = preprocess_function_kwargs_fn()

def predict(test_dataset, namefn):        
    predict_dataset = test_dataset.map(
                    preprocess_function,
                    fn_kwargs=preprocess_function_kwargs,
                    batched=True,
                    num_proc=None,
                    remove_columns=test_dataset.column_names,
                    load_from_cache_file=False,
                    desc="Running tokenizer on prediction dataset",
                )

    predict_dataset = predict_dataset.with_format('torch')
    print(predict_dataset)
    outputs = []
    for ids, attm in zip(predict_dataset["input_ids"], predict_dataset["attention_mask"]):
        with torch.no_grad():
            generated_output = model.generate(input_ids=ids.reshape(torch.Size([1, ids.shape[0]]))
                                    , attention_mask = attm.reshape(torch.Size([1, attm.shape[0]]))
                                    , return_dict=None)
            decoded = tokenizer.decode(generated_output[0]).lower()
            output = ''
            if 'no' in decoded: output = 'No' 
            if 'yes' in decoded: output = 'Yes' 
            outputs.append(output)
        
    print(outputs)
    with open(gen_outputs_path + namefn + '.csv', 'w') as f: 
        f.write('\n'.join(outputs))


OUT_PATH  = '/ukp-storage-1/kuczykowska/data/daic/created_datasets/classification_binary_dataset'#_new_labels'    
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
avrgs = ['macro', 'micro', 'weighted', 'binary']

def evaluate_predictions(path, ds):
    print('Evaluating predictions on ' + ds + ' dataset')
    labels = read_labels(path, True)
    y_p = pd.read_csv(gen_outputs_path + md_name + '_' + ds + '.csv', header = None)
    y_p = [1 if val.lower() == 'yes' else 0 for val in y_p[0].values]
    for avrg in avrgs:
        print(avrg, ':', f1_score(labels.PHQ8_Binary.values, y_p, average = avrg))


evaluate_predictions(TEST_LABEL_PATH, 'test')
evaluate_predictions(DEV_LABEL_PATH, 'dev')
