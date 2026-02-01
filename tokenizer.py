import sentencepiece as spm
import os

def train_tokenizer(files, vocab_size=32000):
    os.makedirs("tokenizer", exist_ok=True)
    spm.SentencePieceTrainer.Train(
        input=",".join(files),
        model_prefix="tokenizer/spm",
        vocab_size=vocab_size
    )
    return "tokenizer/spm.model"
