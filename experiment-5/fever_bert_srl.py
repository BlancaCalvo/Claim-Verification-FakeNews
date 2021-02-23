import numpy as np
import random
import time
import datetime
import argparse
import logging
import json

from allennlp.predictors import Predictor

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup

# FROM SemBERT
from tagged_features import InputExample, convert_examples_to_features, transform_tag_features
from tag_model.tag_tokenization import TagTokenizer
from tag_model.modeling import TagConfig
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForSequenceClassificationTag
from pytorch_pretrained_bert.tokenization import BertTokenizer

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

tokenizer = BertTokenizer.from_pretrained(model_checkpoint, do_lower_case=True)

#predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.11.19.tar.gz")

train_encoded_dataset = convert_examples_to_features(train_dataset, max_seq_length=300, tokenizer=tokenizer, srl_predictor=None)
dev_encoded_dataset = convert_examples_to_features(dev_dataset, max_seq_length=300, tokenizer=tokenizer, srl_predictor=None)
tag_tokenizer = TagTokenizer()

num_labels = 3
max_num_aspect = 3
vocab_size = len(tag_tokenizer.ids_to_tags)
tag_config = TagConfig(tag_vocab_size=vocab_size,
                           hidden_size=10,
                           layer_num=1,
                           output_dim=10,
                           dropout_prob=0.1,
                           num_aspect=max_num_aspect)
#cache_dir = args.cache_dir if args.cache_dir else os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, 'distributed_{}'.format(args.local_rank))
model = BertForSequenceClassificationTag.from_pretrained(model_checkpoint,
              cache_dir=PYTORCH_PRETRAINED_BERT_CACHE,
              num_labels = num_labels,tag_config=tag_config)

def flat_accuracy(preds, labels): # from https://medium.com/@aniruddha.choudhury94/part-2-bert-fine-tuning-tutorial-with-pytorch-for-text-classification-on-the-corpus-of-linguistic-18057ce330e1
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# CREATE TENSOR DATASET
train_features = transform_tag_features(3, train_encoded_dataset, tag_tokenizer, max_seq_length=300) #max_num_aspect=3, check this
all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
all_start_end_idx = torch.tensor([f.orig_to_token_split_idx for f in train_features], dtype=torch.long)
all_input_tag_ids = torch.tensor([f.input_tag_ids for f in train_features], dtype=torch.long)
train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_end_idx, all_input_tag_ids, all_label_ids)

eval_features = transform_tag_features(3, dev_encoded_dataset, tag_tokenizer, max_seq_length=300)
all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
all_start_end_idx = torch.tensor([f.orig_to_token_split_idx for f in eval_features], dtype=torch.long)
all_input_tag_ids = torch.tensor([f.input_tag_ids for f in eval_features], dtype=torch.long)
eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_end_idx, all_input_tag_ids, all_label_ids)

# DATALOADER
batch_size = 16
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)# Create the DataLoader for our validation set.
validation_sampler = SequentialSampler(eval_data)
validation_dataloader = DataLoader(eval_data, sampler=validation_sampler, batch_size=batch_size)

# PARAMETERS
seed_val = 1995
global_step = 0
nb_tr_steps = 0
tr_loss = 0
best_epoch = 0
best_result = 0.0
cuda = 0
epochs = 4
learning_rate = 2e-5
device = torch.device("cuda")

optimizer = AdamW(model.parameters(),lr=learning_rate, eps=1e-8)
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
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
        if step % 40 == 0 and not step == 0: # Progress update every 40 batches.
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, start_end_idx, input_tag_ids, label_ids = batch
        loss = model(input_ids, segment_ids, input_mask, start_end_idx, input_tag_ids, label_ids) #in previous model it was outputs[0]

        tr_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # check this line
        optimizer.step()
        scheduler.step()
    avg_train_loss = tr_loss / len(train_dataloader)

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
        input_ids, input_mask, segment_ids, start_end_idx, input_tag_ids, label_ids = batch

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, start_end_idx, input_tag_ids, None)

        # GET THE LOGITS
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1
    print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))
print("")
print("Training complete!")