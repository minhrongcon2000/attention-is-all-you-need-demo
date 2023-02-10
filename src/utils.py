import torch
import torchtext.transforms as T

from torchtext.data.utils import get_tokenizer
from torchmetrics.functional import bleu_score


def get_sentence_preprocessor_info():
    src_vocab_transform = T.VocabTransform(torch.load("models/src_vocab.pth"))
    tgt_vocab_transform = T.VocabTransform(torch.load("models/tgt_vocab.pth"))

    src_transform = T.Sequential(
        T.Truncate(100),
        T.AddToken(token="<bos>", begin=True),
        T.AddToken(token="<eos>", begin=False),
        src_vocab_transform,
        T.ToTensor(padding_value=1)
    )
    tgt_transform = T.Sequential(
        T.Truncate(100),
        T.AddToken(token="<bos>", begin=True),
        T.AddToken(token="<eos>", begin=False),
        tgt_vocab_transform,
        T.ToTensor(padding_value=1)
    )

    tgt_tokenizer = get_tokenizer("spacy", "en_core_web_sm")
    src_tokenizer = get_tokenizer("spacy", "de_core_news_sm")

    def apply_transform(x):
        src, tgt = x["src"], x["tgt"]
        
        return src_transform(src), tgt_transform(tgt)
    
    return dict(src_transform=src_transform, 
                tgt_transform=tgt_transform,
                src_tokenizer=src_tokenizer,
                tgt_tokenizer=tgt_tokenizer,
                sentence_preprocessor=apply_transform)
    
def translate_sentence(input_sentence_tensor, 
                       model, 
                       max_len=100):
    with torch.no_grad():
        tgt_vocab_transform = torch.load("models/tgt_vocab.pth")
        outputs = ["<bos>"]
        for _ in range(max_len):
            tgt_tensor = torch.LongTensor(tgt_vocab_transform(outputs)).unsqueeze(0)
            pred = model(input_sentence_tensor, tgt_tensor)
            pred_token = pred.argmax(dim=-2)[0, -1]
            
            if pred_token == tgt_vocab_transform.get_stoi()["<eos>"]:
                break
            
            outputs.append(tgt_vocab_transform.get_itos()[pred_token])
            
        
    return " ".join(outputs[1:])

def calc_bleu_score(input_sentence_tensor, tgt_sentence_tensor, model):
    tgt_vocab_transform = torch.load("models/tgt_vocab.pth")
    preds = []
    tgt_sentences = []
    for i in range(len(input_sentence_tensor)):
        src = input_sentence_tensor[i].unsqueeze(0)
        tgt = tgt_sentence_tensor[i]
        pred = translate_sentence(src, model)
        tgt_sentence = " ".join(tgt_vocab_transform.vocab.get_itos()[token] for token in tgt)
        preds.append(pred)
        tgt_sentences.append([tgt_sentence])
        
    return bleu_score(preds, tgt_sentences)
        