from transformers import (
    DistilBertConfig,
    DistilBertForSequenceClassification,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from tokenizers.pre_tokenizers import BertPreTokenizer

from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score

# 1. Data
texts = ["true", "false"]
labels = [1, 0]

# 2. Tokenizer (WordLevel from scratch)
tokenizer_model = models.WordPiece(vocab={}, unk_token="[UNK]")
tokenizer = Tokenizer(tokenizer_model)
tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

tokenizer.train_from_iterator(
    texts,
    trainer=trainers.WordPieceTrainer(
        vocab_size=100, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
    ),
)

tokenizer.decoder = decoders.WordPiece(prefix="##")
tokenizer.post_processor = processors.TemplateProcessing(
    single="[CLS] $A [SEP]",
    special_tokens=[
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ],
)

hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
)

hf_tokenizer.save_pretrained("/tmp")


# 3. Tokenize
def tokenize(example):
    enc = hf_tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=8,
        add_special_tokens=True,
    )
    enc["label"] = example["label"]
    return enc


dataset = Dataset.from_dict({"text": texts, "label": labels}).map(tokenize)

# 4. Model
config = DistilBertConfig(
    vocab_size=hf_tokenizer.vocab_size,
    max_position_embeddings=32,
    n_layers=2,
    dim=64,
    hidden_dim=128,
    n_heads=2,
    num_labels=2,
)
model = DistilBertForSequenceClassification(config)


# 5. Accuracy metric
def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}


# 6. Trainer
args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=50,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=5e-3,
    evaluation_strategy="epoch",
    report_to="none",
    logging_dir="./logs",
    disable_tqdm=True,
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    eval_dataset=dataset,
    compute_metrics=compute_metrics,
)

# 7. Train + Evaluate
trainer.train()
metrics = trainer.evaluate()
print(f"✅ Accuracy: {metrics['eval_accuracy'] * 100:.2f}%")
