from datasets import load_metric, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np

task = "mnli"
model_checkpoint = "bert-base-uncased"
batch_size = 16
actual_task = "mnli"
metric = load_metric('accuracy', actual_task)


def read_examples(input_file):
    """Read input to dictionary."""
    unique_id = 0
    indexes = []
    labels = []
    claims = []
    ids = []
    evidences_list = []
    with open(input_file, "r", encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            line = line.strip().split('\t')

            index = line[0]
            label = line[1]
            claim = line[2]
            evidences = line[3:]
            evi_concat = ' '.join(evidences)

            if label == 'SUPPORTS':
                label = 0
            elif label == 'REFUTES':
                label = 1
            elif label == 'NOTENOUGHINFO':
                label = 2

            indexes.append(index)
            labels.append(label)
            claims.append(claim)
            evidences_list.append(evi_concat)
            ids.append(unique_id)
            unique_id += 1
        examples_dict = {'index': indexes, 'unique_id': ids, 'label': labels, 'text_a': claims,
                         'text_b': evidences_list}
    return examples_dict

train_raw_dataset = read_examples('data/gear/gear-train-set-0_001.tsv')
train_dataset = Dataset.from_dict(train_raw_dataset)
dev_raw_dataset = read_examples('data/gear/gear-dev-set-0_001.tsv')
dev_dataset = Dataset.from_dict(dev_raw_dataset)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

def preprocess_function(examples):
    return tokenizer(examples['text_a'], examples['text_b'], truncation=True)

train_encoded_dataset = train_dataset.map(preprocess_function, batched=True)
dev_encoded_dataset = dev_dataset.map(preprocess_function, batched=True)

num_labels = 3
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

metric_name = "accuracy"

args = TrainingArguments(
    "test-fever",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model,
    args,
    train_dataset=train_encoded_dataset,
    eval_dataset=dev_encoded_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.evaluate()