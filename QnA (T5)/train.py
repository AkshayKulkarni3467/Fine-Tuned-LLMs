from dataloader import return_dataloader,make_df,MODEL_NAME
from model import BioQAModel
from pathlib import Path
from transformers import T5Tokenizer
from lightning.pytorch import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pytorch_lightning as pl

EPOCHS = 6

factoid_paths = sorted(list(Path('data/QA/BioASQ/').glob('BioASQ-train-factoid-*')))

df = make_df(factoid_paths=factoid_paths)
tokenizer = tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
data_module = return_dataloader(df,tokenizer,batch_size = 6)
model = BioQAModel()

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath='checkpoints',
    filename = 'best_checkpoint',
    save_top_k = 1,
    verbose = 1,
    monitor = 'val_loss',
    mode = 'min'
)

trainer = pl.Trainer(callbacks=[checkpoint_callback],
                  max_epochs=EPOCHS,
                  accelerator='gpu',
                  devices=1,
                  num_sanity_val_steps=0
                  )

if __name__ == '__main__':
    trainer.fit(model,data_module)