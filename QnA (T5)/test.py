from model import BioQAModel
from train import tokenizer,df
import torch
import random

trained_model = BioQAModel.load_from_checkpoint('checkpoints/best_checkpoint.ckpt')
trained_model.freeze()

def generate_answer(question):
    source_encoding = tokenizer(
        question['question'],
        question['context'],
        max_length=396,
        padding='max_length',
        truncation='only_second',
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors = 'pt'
    ).to('cuda' if torch.cuda.is_available() else 'cpu')
    generated_ids = trained_model.model.generate(
        input_ids=source_encoding['input_ids'],
        attention_mask=source_encoding['attention_mask'],
        num_beams=2,
        max_length=80,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True,
        use_cache=True
    )
    
    preds = [
        tokenizer.decode(generated_id,skip_special_tokens=True,clean_up_tokenization_spaces=True)
        for generated_id in generated_ids
    ]
    
    return "".join(preds)

if __name__ == '__main__':
    random_num = random.randint(0,6000)
    sample_question = df.iloc[random_num]
    question = sample_question['question']
    original_answer = sample_question['answer_text']
    
    predicted_answer = generate_answer(sample_question)
    
    print(f'Question: {question}\nOriginal ans: {original_answer}, Predicted ans : {predicted_answer}')

