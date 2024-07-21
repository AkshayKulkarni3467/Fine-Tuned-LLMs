from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer
    

ds = load_dataset('tomaarsen/conllpp')
model_checkpoint = 'distilbert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    
def create_tag_names(batch):
    global ds
    tags = ds['train'].features['ner_tags'].feature
    temp = {'ner_tags_str': [tags.int2str(idx) for idx in batch['ner_tags']]}
    return temp
    
def align_labels_with_tokens(labels,word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            new_labels.append(-100)
        else:
            label = labels[word_id]
            if label%2==1:
                label=label+1
            new_labels.append(label)
    return new_labels
    
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples['tokens'],truncation=True,is_split_into_words=True)
    all_labels = examples['ner_tags']
    new_labels = []
    for i,labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels,word_ids))
    tokenized_inputs['labels'] = new_labels
    return tokenized_inputs

    
def mapper():
    global ds
    ds = ds.map(create_tag_names)
    tokenized_datasets = ds.map(tokenize_and_align_labels, batched=True,remove_columns=ds['train'].column_names)
    return ds,tokenized_datasets