from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import numpy as np
from run_classifier import InputExample, convert_examples_to_features, transform_tag_features
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import random
import time
import datetime
from allennlp.predictors import Predictor
import argparse
import logging
import json
from tag_model.tag_tokenization import TagTokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def read_srl_examples(input):
    with open(input, "r") as read_file:
        data = json.load(read_file)
    examples = []
    for element in data:
        examples.append(InputExample(guid=element['unique_id'], text_a=element['claim_srl'], text_b=element['evidence_srl'],label=element['label'], index=element['index']))
    return examples

parser = argparse.ArgumentParser()
parser.add_argument("--train_srl_file", default=None, type=str, required=True)
parser.add_argument("--dev_srl_file", default=None, type=str, required=True)
parser.add_argument("--cuda", default=-1, type=int, required=False) # set to 0
args = parser.parse_args()

model_checkpoint = "bert-base-uncased"

logger.info('Loading srl.')

train_dataset = read_srl_examples(args.train_srl_file)
dev_dataset = read_srl_examples(args.dev_srl_file)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

#predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.11.19.tar.gz")

train_encoded_dataset = convert_examples_to_features(train_dataset, max_seq_length=300, tokenizer=tokenizer, srl_predictor=None)
dev_encoded_dataset = convert_examples_to_features(dev_dataset, max_seq_length=300, tokenizer=tokenizer, srl_predictor=None)
tag_tokenizer = TagTokenizer()

num_labels = 3
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

def flat_accuracy(preds, labels): # from https://medium.com/@aniruddha.choudhury94/part-2-bert-fine-tuning-tutorial-with-pytorch-for-text-classification-on-the-corpus-of-linguistic-18057ce330e1
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# CREATE TENSORS <---- I AM HERE!
train_features = transform_tag_features(3, train_encoded_dataset, tag_tokenizer, max_seq_length=300) #max_num_aspect=3, check this
dev_features = transform_tag_features(3, dev_encoded_dataset, tag_tokenizer, max_seq_length=300)

train_inputs = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
validation_inputs = torch.tensor([f.input_ids for f in dev_features], dtype=torch.long)
train_labels = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
validation_labels = torch.tensor([f.label_id for f in dev_features], dtype=torch.long)
train_masks = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
validation_masks = torch.tensor([f.input_mask for f in dev_features], dtype=torch.long)

# DATALOADER
batch_size = 32
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)# Create the DataLoader for our validation set.
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

# PARAMETERS
seed_val = 42
epochs = 4
cuda = 0
device = torch.device("cuda")
#n_gpu = torch.cuda.device_count()
optimizer = AdamW(model.parameters(),lr=2e-5, eps=1e-8)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
model.to(device)

# TRAINING LOOP
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
loss_values = []
for epoch_i in range(0, epochs):
    # TRAINING
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')
    t0 = time.time()
    total_loss = 0
    model.train()
    for step, batch in enumerate(train_dataloader):
        if step % 40 == 0 and not step == 0: # Progress update every 40 batches.
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        model.zero_grad()
        outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)

        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    avg_train_loss = total_loss / len(train_dataloader)

    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

    # VALIDATION
    print("Running Validation...")
    t0 = time.time()
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask)

        # GET THE LOGITS
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1
    print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))
print("")
print("Training complete!")