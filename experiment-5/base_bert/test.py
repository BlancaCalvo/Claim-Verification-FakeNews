
import argparse
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import numpy as np
from extractor import InputExample, convert_examples_to_features
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler
import random
import os

def read_examples(input_file):
    """Read input to dictionary."""
    unique_id = 0
    examples = []
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

            examples.append(InputExample(unique_id=unique_id, text_a=claim, text_b=evi_concat,
                                         label=label, index=index, is_claim=False))

            unique_id += 1
    return examples

parser = argparse.ArgumentParser()
#
parser.add_argument("--dev_features", default='data/gear/N_gear-dev-set-0_001.tsv', type=str, required=False)

#parser.add_argument("--concat", action='store_true', help="Set this flag if you want to concat evidences.")
#parser.add_argument("--aggregate", action='store_true', help="Set this flag if you want to aggregate the evidences.") #does not work yet
#parser.add_argument("--vote", action='store_true', help="Set this flag if you want to make voting system with evidences.")

parser.add_argument("--seq_length", default=300, type=int, required=False)
parser.add_argument("--batch_size", default=16, type=int, required=False)
#parser.add_argument("--cuda_devices", default='-1', type=str, required=False)

args = parser.parse_args()
#args.cuda = not args.no_cuda and torch.cuda.is_available()
seed = 1995
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

dev_dataset = read_examples(args.dev_features)

model_checkpoint = "bert-base-uncased"
#tokenizer = BertTokenizer.from_pretrained(model_checkpoint, do_lower_case=True)
tokenizer = BertTokenizer.from_pretrained(model_checkpoint, use_fast=True)

#logger.info('Convert examples to features (VALIDATION).')
dev_encoded_dataset = convert_examples_to_features(examples=dev_dataset, seq_length=250, tokenizer=tokenizer)
num_labels = 3
model = BertForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

def flat_accuracy(preds, labels): # from https://medium.com/@aniruddha.choudhury94/part-2-bert-fine-tuning-tutorial-with-pytorch-for-text-classification-on-the-corpus-of-linguistic-18057ce330e1
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# CREATE TENSORS
#train_inputs = torch.tensor([f.input_ids for f in train_encoded_dataset], dtype=torch.long)
validation_inputs = torch.tensor([f.input_ids for f in dev_encoded_dataset], dtype=torch.long)
#train_labels = torch.tensor([f.label for f in train_encoded_dataset], dtype=torch.long)
validation_labels = torch.tensor([f.label for f in dev_encoded_dataset], dtype=torch.long)
#train_masks = torch.tensor([f.input_mask for f in train_encoded_dataset], dtype=torch.long)
validation_masks = torch.tensor([f.input_mask for f in dev_encoded_dataset], dtype=torch.long)

# DATALOADER
batch_size = 16
#train_data = TensorDataset(train_inputs, train_masks, train_labels)
#train_sampler = RandomSampler(train_data)
#train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)# Create the DataLoader for our validation set.
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

# PARAMETERS
seed_val = 42
epochs = 4
cuda = 0
device = torch.device("cuda")
#n_gpu = torch.cuda.device_count()
#optimizer = AdamW(model.parameters(),lr=2e-5, eps=1e-8)
#total_steps = len(train_dataloader) * epochs
#scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
model.to(device)
best_epoch = 0
best_result = 0.0

seeds = [314]

for seed in seeds:
    base_dir = 'experiment-5/outputs/F-base-bert-2/'

    # FOR SOME REASON KEYS OF THE DICT CONTAIN MODULE., CHECK WHY
    checkpoint = torch.load(base_dir + 'best.pth.tar')
    #print(type(checkpoint['model'])) # should be dict
    # new_checkpoint = checkpoint
    # checkpoint_new_names = {}
    # for i in checkpoint['model'].keys():
    #     checkpoint_new_names[i] = i[7:]
    # new_checkpoint['model'] = dict((checkpoint_new_names[key], value) for (key, value) in checkpoint['model'].items())

    model.load_state_dict(checkpoint['model'])
    model.eval()

    fout = open(base_dir + 'dev-results.tsv', 'w')
    #dev_tqdm_iterator = tqdm(dev_dataloader)
    with torch.no_grad():
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
            #eval_accuracy += tmp_eval_accuracy
            #nb_eval_steps += 1

            for i in range(logits.shape[0]):
                #fout.write('\t'.join(['%.4lf' % num for num in logits[i]]) + '\r\n')
                fout.write('{}\t{}\t{}\t{}\n'.format(logits[i][0], logits[i][1], logits[i][2], label_ids[i]))
    fout.close()

    # fout = open(base_dir + 'test-results.tsv', 'w')
    # test_tqdm_iterator = tqdm(test_dataloader)
    # with torch.no_grad():
    #     for index, data in enumerate(test_tqdm_iterator):
    #         feature_batch, claim_batch = data
    #         feature_batch = feature_batch.cuda()
    #         claim_batch = claim_batch.cuda()
    #         outputs = model(feature_batch, claim_batch)
    #
    #         for i in range(outputs.shape[0]):
    #             fout.write('\t'.join(['%.4lf' % num for num in outputs[i]]) + '\r\n')
    # fout.close()
