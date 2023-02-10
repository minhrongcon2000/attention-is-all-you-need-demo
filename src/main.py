import argparse

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k

from transformer import AttentionV1
from utils import get_sentence_preprocessor_info

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--wandb_api_key", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=32)
args = vars(parser.parse_args())

preprocessor_info = get_sentence_preprocessor_info()

train_dataset = Multi30k(root="data", split="train").map(lambda x: (preprocessor_info["src_tokenizer"](x[0]), preprocessor_info["tgt_tokenizer"](x[1])))\
                                                    .batch(args["batch_size"])\
                                                    .rows2columnar(["src", "tgt"])\
                                                    .map(preprocessor_info["sentence_preprocessor"])
                                              
val_dataset = Multi30k(root="data", split="dev").map(lambda x: (preprocessor_info["src_tokenizer"](x[0]), preprocessor_info["tgt_tokenizer"](x[1])))\
                                                .batch(args["batch_size"])\
                                                .rows2columnar(["src", "tgt"])\
                                                .map(preprocessor_info["sentence_preprocessor"])

train_dataloaders = DataLoader(train_dataset, batch_size=None, shuffle=True)
val_dataloaders = DataLoader(val_dataset, batch_size=None)

model = AttentionV1(len(preprocessor_info["src_transform"][3].vocab), 
                    len(preprocessor_info["tgt_transform"][3].vocab), 
                    512, 
                    8)

logger = WandbLogger(project="AttentionWMT",
                     name="Attention")

trainer = Trainer(accelerator="gpu",
                  max_epochs=args["epochs"],
                  logger=logger,
                  enable_progress_bar=False)

trainer.fit(model=model, 
            train_dataloaders=train_dataloaders, 
            val_dataloaders=val_dataloaders)

