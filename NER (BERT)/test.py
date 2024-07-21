from transformers import pipeline
import torch

trained_model = './model/checkpoint-5268'

ner_tagger = pipeline(
    'token-classification', model=trained_model,aggregation_strategy='simple',device = 'cuda' if torch.cuda.is_available() else 'cpu'
)

outputs = ner_tagger(input('Enter a sentence : \n'))

info = [(i['word'],i['entity_group']) for i in outputs]

print(info)