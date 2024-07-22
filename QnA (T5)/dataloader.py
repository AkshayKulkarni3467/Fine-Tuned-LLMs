import json
import pandas as pd
from torch.utils.data import Dataset,DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from pathlib import Path
from transformers import T5Tokenizer

MODEL_NAME = 't5-base'

def extract_qna(factoid_path:Path):
    with factoid_path.open() as json_file:
        data = json.load(json_file)
    questions = data['data'][0]['paragraphs']
    data_rows = []
    
    for question in questions:
        context = question['context']
        for question_and_ans in question['qas']:
            question = question_and_ans['question']
            answers = question_and_ans['answers']
            for answer in answers:
                answer_text = answer['text']
                answer_start = answer['answer_start']
                answer_end = answer_start + len(answer_text)
                
                data_rows.append({
                    'question':question,
                    'context':context,
                    'answer_text':answer_text,
                    'answer_start':answer_start,
                    'answer_end':answer_end
                })
    return pd.DataFrame(data_rows)


def make_df(factoid_paths):
    dfs = []

    for factoid_path in factoid_paths:
        dfs.append(extract_qna(factoid_path=factoid_path))

    df = pd.concat(dfs)
    df = df.drop_duplicates(subset=['context']).reset_index(drop=True)
    
    return df

class BioQADataset(Dataset):
    def __init__(
        self,
        data:pd.DataFrame,
        tokenizer: T5Tokenizer,
        source_max_token_len: int = 396,
        target_max_token_len: int = 32
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = data
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index:int):
        data_row = self.data.iloc[index]
        
        source_encoding = self.tokenizer(
            data_row['question'],
            data_row['context'],
            max_length = self.source_max_token_len,
            padding = 'max_length',
            truncation = 'only_second',
            return_attention_mask = True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        
        target_encoding = self.tokenizer(
            data_row['answer_text'],
            max_length=self.target_max_token_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        
        labels = target_encoding['input_ids']
        labels[labels == 0] = -100
        
        return dict(
            question=data_row['question'],
            context = data_row['context'],
            answer_text = data_row['answer_text'],
            input_ids = source_encoding['input_ids'].flatten(),
            attention_mask = source_encoding['attention_mask'].flatten(),
            labels = labels.flatten()
        )
        

class BioQADataModule(pl.LightningDataModule):
    def __init__(
        self,
        df : pd.DataFrame,
        tokenizer: T5Tokenizer,
        batch_size: int = 6,
        source_max_token_len: int = 396,
        target_max_token_len: int = 32
    ):
        super().__init__()
        self.batch_size  = batch_size
        self.source_token_max_len = source_max_token_len
        self.target_token_max_len = target_max_token_len
        self.tokenizer = tokenizer
        train_df,test_df = train_test_split(df,test_size=0.05)
        self.train_df = train_df
        self.test_df = test_df
    
    def setup(self,stage=None):
        self.train_dataset = BioQADataset(
            self.train_df,
            self.tokenizer,
            self.source_token_max_len,
            self.target_token_max_len,
        )
        
        self.test_dataset = BioQADataset(
            self.test_df,
            self.tokenizer,
            self.source_token_max_len,
            self.target_token_max_len
        )
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            shuffle=True,
            num_workers=1,
            persistent_workers=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size = 1
        )
        
def return_dataloader(df,tokenizer,batch_size):
    
    data_module = BioQADataModule(df,tokenizer,batch_size=batch_size)
    data_module.setup()
    return data_module