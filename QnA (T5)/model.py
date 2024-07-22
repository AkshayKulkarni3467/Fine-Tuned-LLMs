from transformers import AdamW,T5ForConditionalGeneration
from dataloader import MODEL_NAME
import pytorch_lightning as pl

class BioQAModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME,return_dict = True)
    
    def forward(self,input_ids,attention_mask,labels =None):
        output = self.model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels = labels
        )
        return output.loss, output.logits
    
    def training_step(self,batch,batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log('train_loss',loss,prog_bar=True,logger=True)
        return loss
    
    def validation_step(self,batch,batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss,outputs = self(input_ids,attention_mask,labels)
        self.log('val_loss',loss,prog_bar=True,logger=True)
        return loss
    
    def test_step(self,batch,batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss,outputs = self(input_ids,attention_mask,labels)
        self.log('test_loss',loss,prog_bar=True,logger=True)
        return loss
    
    def configure_optimizers(self):
        return AdamW(self.parameters(),lr = 1e-4)

    
    
    
    
    
        
        
        
        
        
        
        