from transformers import DataCollatorForTokenClassification
from build_dataset import mapper,tokenizer,model_checkpoint
import evaluate
import numpy as np
from transformers import AutoModelForTokenClassification,TrainingArguments,Trainer




def get_labels(ds):
    return ds['train'].features['ner_tags'].feature.names

def compute_metric(eval_preds):
    logits,labels= eval_preds
    metric = evaluate.load('seqeval')
    predictions = np.argmax(logits,axis=-1)
    label_names = get_labels(ds)
    
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [[label_names[p] for p,l in zip(prediction,label) if l !=-100] for prediction,label in zip(predictions,labels)]
    
    all_metrics = metric.compute(predictions=true_predictions,references=true_labels)
    
    return {'precision' : all_metrics['overall_precision'],
            'recall' : all_metrics['overall_recall'],
            'f1' : all_metrics['overall_f1'],
            'accuracy' : all_metrics['overall_accuracy']}
    
def get_params(ds):
    label_names = get_labels(ds)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    id2label = {i:label for i,label in enumerate(label_names)}
    label2id = {label:i for i,label in enumerate(label_names)}
    return data_collator,id2label,label2id

def get_model(ds):
    _,id2label,label2id = get_params(ds)
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint,id2label=id2label,label2id=label2id)
    return model


def get_trainer(ds,tokenized_dataset,num_of_epochs,lr,weight_decay):
    model = get_model(ds)
    data_collator,_,_ = get_params(ds)
    args = TrainingArguments('model',
                             evaluation_strategy='epoch',
                             save_strategy='epoch',
                             learning_rate=lr,
                             num_train_epochs=num_of_epochs,
                             weight_decay=weight_decay)


    trainer = Trainer(model=model,
                      args=args,
                      train_dataset=tokenized_dataset['train'],
                      eval_dataset=tokenized_dataset['validation'],
                      data_collator=data_collator,
                      compute_metrics=compute_metric,
                      tokenizer=tokenizer)
    return trainer

if __name__ == '__main__':
    ds,tokenized_dataset = mapper()
    num_of_epochs = 3
    lr = 1e-4
    weight_decay = 0.01
    trainer = get_trainer(ds,tokenized_dataset,num_of_epochs,lr,weight_decay)
    trainer.train()