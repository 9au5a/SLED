from sled import SledTokenizer, SledModel
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
import torch
from run import preprocess_function
from copy import deepcopy
from sklearn.metrics import f1_score

import pandas as pd

PREFIX_DOC_SEP = '\n\n'
max_source_length = 16384


md_name = 'D8'
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
            "input_prefix_column": '', # todo delete this
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
                    desc="Running tokenizer on " + namefn + " dataset",
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
            if 'not' in decoded: output = 'Not' 
            if 'depressed' in decoded: output = 'Depressed' 
            outputs.append(output)
        
    print(outputs)
    with open(gen_outputs_path + namefn + '.csv', 'w') as f: 
        f.write('\n'.join(outputs))


OUT_PATH  = '/ukp-storage-1/kuczykowska/data/depressometer/created_datasets/classification_binary_dataset'#_new_labels'    
test_dataset = load_dataset(OUT_PATH, split = 'test', cache_dir='./', )
test_labels = test_dataset['output']
print(test_labels)
predict(test_dataset, md_name + '_test')
test_dataset = load_dataset(OUT_PATH, split = 'validation', cache_dir='./', )
dev_labels = test_dataset['output']
predict(test_dataset, md_name + '_dev')

avrgs = ['macro', 'micro', 'weighted', 'binary']

def evaluate_predictions(y, ds):
    print('Evaluating predictions on ' + ds + ' dataset')
    y_p = pd.read_csv(gen_outputs_path + md_name + '_' + ds + '.csv', header = None)
    y_p = [1 if val.lower() == 'depressed' else 0 for val in y_p[0].values]
    y = [1 if val.lower() == 'depressed' else 0 for val in y]
    for avrg in avrgs:
        print(avrg, ':', f1_score(y, y_p, average = avrg))


evaluate_predictions(test_labels, 'test')
evaluate_predictions(dev_labels, 'dev')
