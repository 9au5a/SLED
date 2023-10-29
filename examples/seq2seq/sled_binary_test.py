from sled import SledTokenizer, SledModel
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
import torch

'''
#  start ----------------------------- regression 
DS_PATH  = '/ukp-storage-1/kuczykowska/data/daic/created_datasets/regression_dataset'    
test_dataset = load_dataset(DS_PATH, split = 'validation', cache_dir='./', )

tokenizer_path = '/ukp-storage-1/kuczykowska/code/SLED/tmp/daic_regression/config.json'
tokenizer = SledTokenizer.from_pretrained(tokenizer_path)

path = '/ukp-storage-1/kuczykowska/code/SLED/tmp/daic_regression/'
model = AutoModelForSeq2SeqLM.from_pretrained(path)

#from run import _preprocess_raw_inputs
from run import preprocess_function

PREFIX_DOC_SEP = '\n\n'
from copy import deepcopy
max_target_length = 16384
def preprocess_function_kwargs_fn():
        return {
            "tokenizer": deepcopy(tokenizer),
            "prefix": '',
            "input_column": 'input',
            "input_prefix_column": 'input_prefix',
            "output_column": 'output',
            "max_source_length": max_target_length,
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
preprocess_function_kwargs["max_target_length"] = max_target_length
        
predict_dataset = test_dataset.map(
                preprocess_function,
                fn_kwargs=preprocess_function_kwargs,
                batched=True,
                num_proc=None,
                remove_columns=test_dataset.column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on prediction dataset",
            )
outputs = []
predict_dataset = predict_dataset.with_format('torch')
for ids, prefl in zip(predict_dataset["input_ids"], predict_dataset["prefix_length"]):
    #print(ids, prefl)
    with torch.no_grad():
        generated_output = model.generate(input_ids=ids
                                        , prefix_length=prefl
                                        #, todo try it attention_mask
                                        , return_dict=None)
        decoded = tokenizer.decode(generated_output[0]).lower()
        outputs.append(decoded)
    
print(outputs)


exit()


path = '/ukp-storage-1/kuczykowska/code/SLED/tmp/daic_regression'
model = AutoModelForSeq2SeqLM.from_pretrained(path)



outputs = []
for inp, pref in zip(test_dataset["input"], test_dataset["input_prefix"]):
    test = tokenizer(inp, pref, return_tensors="pt")
    with torch.no_grad():
        generated_output = model.generate(input_ids=test.input_ids, return_dict=None)
        decoded = tokenizer.decode(generated_output[0]).lower()
        output = ''
        if 'no' in decoded: output = 'No' 
        if 'yes' in decoded: output = 'Yes' 
        outputs.append(output)
    
print(outputs)
with open('generated_test_output.csv', 'w') as f: f.write('\n'.join(outputs))
# end ----------------------------- regression 


exit()
# start multi class ------------------------
path = '/ukp-storage-1/kuczykowska/code/SLED/tmp/daic_classification_severity_new_labels'
model = AutoModelForSeq2SeqLM.from_pretrained(path)

# end multi class ------------------------

exit()
'''
#  start ----------------------------- binary 
md_name = 'T4'
#path = '/ukp-storage-1/kuczykowska/code/SLED/tmp/daic_classification_binary_new_labels'
path ='/ukp-storage-1/kuczykowska/code/SLED/tmp/72_dev_70_test_test_daic_classification_binary_new_labels'
path = '/ukp-storage-1/kuczykowska/code/SLED/tmp/test7342_8GPU_batch2_test_daic_classification_binary_new_labels_2'
path = '/ukp-storage-1/kuczykowska/code/SLED/tmp/test706_test_daic_classification_binary_new_labels'
path = '/ukp-storage-1/kuczykowska/code/SLED/tmp/test_daic_classification_binary_new_labels_2'
path = '/ukp-storage-1/kuczykowska/code/SLED/tmp/test_daic_classification_binary_new_labels'
path = '/ukp-storage-1/kuczykowska/code/SLED/tmp/' +  md_name
model = AutoModelForSeq2SeqLM.from_pretrained(path)

#tokenizer_path = '/ukp-storage-1/kuczykowska/code/SLED/tmp/daic_classification_binary_new_labels/config.json'
tokenizer_path = '/ukp-storage-1/kuczykowska/code/SLED/tmp/test_daic_classification_binary_new_labels/config.json'
tokenizer_path = '/ukp-storage-1/kuczykowska/code/SLED/tmp/72_dev_70_test_test_daic_classification_binary_new_labels/config.json'
tokenizer_path = '/ukp-storage-1/kuczykowska/code/SLED/tmp/test7342_8GPU_batch2_test_daic_classification_binary_new_labels_2/config.json'
tokenizer_path = '/ukp-storage-1/kuczykowska/code/SLED/tmp/test706_test_daic_classification_binary_new_labels/config.json'
tokenizer_path = '/ukp-storage-1/kuczykowska/code/SLED/tmp/test_daic_classification_binary_new_labels_2/config.json'
tokenizer_path = '/ukp-storage-1/kuczykowska/code/SLED/tmp/test_daic_classification_binary_new_labels/config.json'
tokenizer_path = '/ukp-storage-1/kuczykowska/code/SLED/tmp/'+ md_name +'/config.json'
tokenizer = SledTokenizer.from_pretrained(tokenizer_path)

''' test z regression tu sie zavczyna '''
from run import preprocess_function

PREFIX_DOC_SEP = '\n\n'
from copy import deepcopy
max_target_length = 16384
def preprocess_function_kwargs_fn():
        return {
            "tokenizer": deepcopy(tokenizer),
            "prefix": '',
            "input_column": 'input',
            "input_prefix_column": 'input_prefix',
            "output_column": 'output',
            "max_source_length": max_target_length,
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
preprocess_function_kwargs["max_target_length"] = max_target_length


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
    '''tu si ekonzcy'''
    outputs = []
    for ids, attm in zip(predict_dataset["input_ids"], predict_dataset["attention_mask"]):
    #for text in test_dataset["input"]:
        #test = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            #generated_output = model.generate(input_ids=test.input_ids
            #                                  , attention_mask = test.attention_mask
            #                                  , return_dict=None)
            generated_output = model.generate(input_ids=ids.reshape(torch.Size([1, ids.shape[0]]))
                                    , attention_mask = attm.reshape(torch.Size([1, attm.shape[0]]))
                                    , return_dict=None)
            decoded = tokenizer.decode(generated_output[0]).lower()
            output = ''
            if 'no' in decoded: output = 'No' 
            if 'yes' in decoded: output = 'Yes' 
            outputs.append(output)
        
    print(outputs)
    with open('/ukp-storage-1/kuczykowska/code/SLED/tmp/generated_outputs/' + namefn + '.csv', 'w') as f: f.write('\n'.join(outputs))
    
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

avrgs = ['macro', 'micro', 'weighted', 'binary']
print('Evaluating predictions on test dataset')
TEST_LABEL_PATH = '/ukp-storage-1/kuczykowska/data/daic/labels/full_test_split.csv' 
labels = read_labels(TEST_LABEL_PATH, True)
y_p = pd.read_csv('/ukp-storage-1/kuczykowska/code/SLED/tmp/generated_outputs/' + md_name + '_test' + '.csv', header = None)
y_p = [1 if val.lower() == 'yes' else 0 for val in y_p[0].values]
for avrg in avrgs:
    print(avrg, ':', f1_score(labels.PHQ8_Binary.values, y_p, average = avrg))

print('Evaluating predictions on development dataset')
DEV_LABEL_PATH = '/ukp-storage-1/kuczykowska/data/daic/labels/dev_split_Depression_AVEC2017.csv' 
labels = read_labels(DEV_LABEL_PATH)
y_p = pd.read_csv('/ukp-storage-1/kuczykowska/code/SLED/tmp/generated_outputs/' + md_name + '_dev' + '.csv', header = None)
y_p = [1 if val.lower() == 'yes' else 0 for val in y_p[0].values]
for avrg in avrgs:
    print(avrg, ':', f1_score(labels.PHQ8_Binary.values, y_p, average = avrg))


# end ----------------------------- binary 


'''
def tokenization(example):
    return tokenizer(example["input"], return_tensors="pt", max_length = 16384, padding = 'max_length')

dataset = test_dataset.map(tokenization, batched=True).with_format("torch")  
#dataset = dataset.with_format("torch")  
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

with torch.no_grad():
    generated_output = model.generate(**next(iter(dataloader)))
print(tokenizer.decode(generated_output)) #encoder_hidden_states

'''