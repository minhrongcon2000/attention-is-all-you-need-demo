import argparse
import torch
import torchtext.functional as F

from collections import OrderedDict
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import Multi30k
from torchtext.vocab import build_vocab_from_iterator, vocab


parser = argparse.ArgumentParser()
parser.add_argument("--max_seq_len", type=int, default=100)
parser.add_argument("--min_freq", type=int, default=3)
args = vars(parser.parse_args())

tgt_lang_model = get_tokenizer("spacy", "en_core_web_sm")
src_lang_model = get_tokenizer("spacy", "de_core_news_sm")


MAX_LEN = args["max_seq_len"]
MIN_FREQ = args["min_freq"]

def apply_transform(x):
    src, tgt = x
    
    src = src_lang_model(src)
    src = F.truncate(src, MAX_LEN)
    
    tgt = tgt_lang_model(tgt)
    tgt = F.truncate(tgt, MAX_LEN)
    
    return src, tgt

def src_extract(x):
    return x[0]

def tgt_extract(x):
    return x[1]

train_datasets = Multi30k(root="data", split="train").map(apply_transform)
dev_datasets = Multi30k(root="data", split="valid").map(apply_transform)

src_train = train_datasets.map(src_extract)
src_dev = dev_datasets.map(src_extract)

tgt_train = train_datasets.map(tgt_extract)
tgt_dev = dev_datasets.map(tgt_extract)

src_train_vocab = build_vocab_from_iterator(src_train, specials=["<unk>"], min_freq=MIN_FREQ)
src_dev_vocab = build_vocab_from_iterator(src_dev, specials=["unk"], min_freq=MIN_FREQ)

src_vocab = set(src_train_vocab.get_stoi().keys()).union(set(src_dev_vocab.get_stoi().keys()))
src_vocab.add("<bos>")
src_vocab.add("<eos>")
src_vocab = vocab(OrderedDict((src_word, 1) for src_word in src_vocab))
src_vocab.set_default_index(src_vocab["<unk>"])

tgt_train_vocab = build_vocab_from_iterator(tgt_train, specials=["<unk>"], min_freq=MIN_FREQ)
tgt_dev_vocab = build_vocab_from_iterator(tgt_dev, specials=["<unk>"], min_freq=MIN_FREQ)

tgt_vocab = set(tgt_train_vocab.get_stoi().keys()).union(set(tgt_dev_vocab.get_stoi().keys()))
tgt_vocab.add("<bos>")
tgt_vocab.add("<eos>")
tgt_vocab = vocab(OrderedDict((tgt_word, 1) for tgt_word in tgt_vocab))
tgt_vocab.set_default_index(tgt_vocab["<unk>"])

torch.save(src_vocab, "models/src_vocab.pth")
torch.save(tgt_vocab, "models/tgt_vocab.pth")
