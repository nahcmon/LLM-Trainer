from trainer import Trainer

if __name__ == "__main__":
    cfg = {
        "dataset": "wikitext",  # Hugging Face dataset name or URL
        "split": "train",
        "seq_length": 1024,
        "vocab_size": 32000,
        "n_layers": 12,
        "hidden_size": 768,
        "n_heads": 12,
        "batch_size": 1,
        "epochs": 1,
        "precision": "fp16",
        "gradient_checkpointing": True
    }

    trainer = Trainer(cfg)
    trainer.train()
