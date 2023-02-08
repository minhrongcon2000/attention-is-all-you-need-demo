import torch
import torch.nn.functional as F
import torchtext.transforms as T

from pytorch_lightning import Trainer
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer

from transformer import AttentionV1


src_vocab_transform = T.VocabTransform(torch.load("models/src_vocab.pth"))
tgt_vocab_transform = T.VocabTransform(torch.load("models/tgt_vocab.pth"))

src_transform = T.Sequential(
    src_vocab_transform,
    T.Truncate(100),
    T.AddToken(token=0, begin=True),
    T.AddToken(token=2, begin=False),
    T.ToTensor(padding_value=1)
)
tgt_transform = T.Sequential(
    tgt_vocab_transform,
    T.Truncate(100),
    T.AddToken(token=0, begin=True),
    T.AddToken(token=2, begin=False),
    T.ToTensor(padding_value=1)
)

src_tokenizer = get_tokenizer("spacy", "en_core_web_sm")
tgt_tokenizer = get_tokenizer("spacy", "de_core_news_sm")

def apply_transform(x):
    src, tgt = x["src"], x["tgt"]
    
    return src_transform(src), tgt_transform(tgt)

dataset = Multi30k(root="data", split="train").map(lambda x: (src_tokenizer(x[0]), tgt_tokenizer(x[1])))\
                                              .batch(32)\
                                              .rows2columnar(["src", "tgt"])\
                                              .map(apply_transform)

model = AttentionV1(len(src_vocab_transform.vocab), len(tgt_vocab_transform.vocab), 512, 8)
trainer = Trainer(accelerator="cpu",
                  max_epochs=1)
trainer.fit(model=model, train_dataloaders=dataset)

