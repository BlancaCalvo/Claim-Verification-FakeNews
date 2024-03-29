import os

import numpy as np
import random
import time
import datetime
import argparse
import logging
import json
import itertools
import operator

#from allennlp.predictors import Predictor

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.models.bert.tokenization_bert import BertTokenizer

# FROM SemBERT
from tagged_features import InputExample, convert_examples_to_features, transform_tag_features
from tag_model.tag_tokenization import TagTokenizer
from tag_model.modeling import TagConfig
from sembert.modeling import BertForSequenceClassificationTag, BertForSequenceClassificationTagWithAgg

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def map_srl(srl, type=None):
    if type == None:
        return srl
    else:
        tags1 = {'[PAD]': '[PAD]', '[CLS]': '[CLS]', '[SEP]': '[SEP]', 'B-V': 'V', 'I-V': 'V', 'B-ARG0': 'ARG0',
                  'I-ARG0': 'ARG0', 'B-ARG1': 'ARG1', 'I-ARG1': 'ARG1', 'B-ARG2': 'ARG2', 'I-ARG2': 'ARG2', 'B-ARG3':'ARG3', 'I-ARG3':'ARG3', 'B-ARG4': 'ARG4',
                  'I-ARG4': 'ARG4', 'B-ARGM-TMP': 'TMP', 'I-ARGM-TMP': 'TMP', 'B-ARGM-LOC': 'LOC', 'I-ARGM-LOC': 'LOC',
                  'B-ARGM-CAU': 'ARGM', 'I-ARGM-CAU': 'ARGM', 'B-ARGM-PRP': 'ARGM',  'B-R-ARG0':'ARG0', 'I-R-ARG0':'ARG0',
                  'I-ARGM-PRP': 'ARGM', 'O': 'O',  'B-ARGM-MNR':'ARGM', 'I-ARGM-MNR':'ARGM', 'B-ARGM-ADV':'ARGM', 'I-ARGM-ADV': 'ARGM',
                    'B-ARGM-DIS':'ARGM', 'I-ARGM-DIS':'ARGM'}
        tags_dream = {'[PAD]': 'PAD', '[CLS]': '[CLS]', '[SEP]': '[SEP]', 'B-V': 'verb', 'I-V': 'verb', 'B-ARG0': 'argument',
                  'I-ARG0': 'argument', 'B-ARG1': 'argument', 'I-ARG1': 'argument', 'B-ARG2': 'argument', 'I-ARG2': 'argument', 'B-ARG3':'argument', 'I-ARG3':'argument', 'B-ARG4': 'argument',
                  'I-ARG4': 'argument', 'B-ARGM-TMP': 'temporal', 'I-ARGM-TMP': 'temporal', 'B-ARGM-LOC': 'location', 'I-ARGM-LOC': 'location',
                  'B-ARGM-CAU': 'argument', 'I-ARGM-CAU': 'argument', 'B-ARGM-PRP': 'argument',  'B-R-ARG0':'argument', 'I-R-ARG0':'argument',
                  'I-ARGM-PRP': 'argument', 'O': 'O',  'B-ARGM-MNR':'argument', 'I-ARGM-MNR':'argument', 'B-ARGM-ADV':'argument', 'I-ARGM-ADV': 'argument',
                    'B-ARGM-DIS':'argument', 'I-ARGM-DIS':'argument'}
        binary = {'[PAD]': 'PAD', '[CLS]': '[CLS]', '[SEP]': '[SEP]', 'B-V': 'verb', 'I-V': 'verb',
                      'B-ARG0': 'argument',
                      'I-ARG0': 'argument', 'B-ARG1': 'argument', 'I-ARG1': 'argument', 'B-ARG2': 'argument',
                      'I-ARG2': 'argument', 'B-ARG3': 'argument', 'I-ARG3': 'argument', 'B-ARG4': 'argument',
                      'I-ARG4': 'argument', 'B-ARGM-TMP': 'argument', 'I-ARGM-TMP': 'argument',
                      'B-ARGM-LOC': 'argument', 'I-ARGM-LOC': 'argument',
                      'B-ARGM-CAU': 'argument', 'I-ARGM-CAU': 'argument', 'B-ARGM-PRP': 'argument',
                      'B-R-ARG0': 'argument', 'I-R-ARG0': 'argument',
                      'I-ARGM-PRP': 'argument', 'O': 'O', 'B-ARGM-MNR': 'argument', 'I-ARGM-MNR': 'argument',
                      'B-ARGM-ADV': 'argument', 'I-ARGM-ADV': 'argument',
                      'B-ARGM-DIS': 'argument', 'I-ARGM-DIS': 'argument'}
        for i,dic in enumerate(srl['verbs']):
            if type=='tags1':
                srl['verbs'][i]['tags'] = list(map(tags1.get, srl['verbs'][i]['tags']))
            elif type == 'dream':
                srl['verbs'][i]['tags'] = list(map(tags_dream.get, srl['verbs'][i]['tags']))
            elif type=='binary':
                srl['verbs'][i]['tags'] = list(map(binary.get, srl['verbs'][i]['tags']))
        return srl

def read_srl_examples(input):
    with open(input, "r") as read_file:
        data = json.load(read_file)
    examples = []
    for element in data:
        examples.append(InputExample(guid=element['unique_id'], text_a=element['claim_srl'], text_b=element['evidence_srl'],label=element['label'], index=element['index']))
    return examples

def read_srl_examples_concat(input, mapping=None):
    with open(input, "r") as read_file:
        data = json.load(read_file)
    examples = []
    res = {'index':''} #'unique_id': None, 'claim_srl':None,'evidence_srl':None,'label':None,
    first = 1
    for dic in data:
        #print(dic['claim_srl'])
        dic['claim_srl'] = map_srl(dic['claim_srl'], mapping)
        dic['evidence_srl'] = map_srl(dic['evidence_srl'], mapping)
        if dic['index'] == res['index']:
            res['evidence_srl']['verbs'] += (dic['evidence_srl']['verbs'])
            res['evidence_srl']['words'] += (dic['evidence_srl']['words'])
        else:
            if first == 1:
                res = dic
                first=0
                continue
            examples.append(InputExample(guid=res['index'], text_a=res['claim_srl'], text_b=res['evidence_srl'],label=res['label'], index=res['index']))
            res = dic
    examples.append(InputExample(guid=res['index'], text_a=res['claim_srl'], text_b=res['evidence_srl'], label=res['label'], index=res['index']))
    return examples

def flat_accuracy(preds, labels): # from https://medium.com/@aniruddha.choudhury94/part-2-bert-fine-tuning-tutorial-with-pytorch-for-text-classification-on-the-corpus-of-linguistic-18057ce330e1
    pred_flat = np.argmax(preds, axis=1)#.flatten()
    labels_flat = labels#.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def top_vote(l):
    values = []
    keys = []
    for k, g in itertools.groupby(l, operator.itemgetter(0)):
        a = [x[1] for x in g]
        value = sorted(a,key=a.count,reverse=True)[0]
        values.append(value)
        keys.append(k)
    return values, keys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default=None, type=str, required=False)
    parser.add_argument("--dev_file", default=None, type=str, required=False)
    parser.add_argument("--batch_size", default=16, type=int, required=False)
    parser.add_argument("--seq_length", default=300, type=int, required=False)
    parser.add_argument("--cuda_devices", default='0', type=str, required=False)
    parser.add_argument("--max_num_aspect", default=3, type=int, required=False)
    parser.add_argument("--mapping", default=None, type=str, required=False)
    #parser.add_argument("--concat", action='store_true', help="Set this flag if you want to concat evidences.") # that's the default, no need for tag
    parser.add_argument("--aggregate", action='store_true', help="Set this flag if you want to aggregate the evidences.") #does not work yet
    parser.add_argument("--vote", action='store_true', help="Set this flag if you want to make voting system with evidences.")
    args = parser.parse_args()

    dir_path = 'outputs/f_sembert-concat_True-agg_%s-%dbatch_size-%dseq_length-%dn_aspect-%s/' % (str(args.aggregate), args.batch_size, args.seq_length, args.max_num_aspect, str(args.mapping))
    logger.info(dir_path)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    model_checkpoint = "bert-base-uncased"

    logger.info('Loading srl.')

    if args.vote:
        train_dataset = read_srl_examples(args.train_file)
        dev_dataset = read_srl_examples(args.dev_file)
    else:
        train_dataset = read_srl_examples_concat(args.train_file, args.mapping)
        dev_dataset = read_srl_examples_concat(args.dev_file, args.mapping)

    tokenizer = BertTokenizer.from_pretrained(model_checkpoint, do_lower_case=True)

    #predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.11.19.tar.gz")

    logger.info('Convert examples to features (TRAIN).')
    train_encoded_dataset = convert_examples_to_features(train_dataset, max_seq_length=args.seq_length, tokenizer=tokenizer, srl_predictor=None)
    logger.info('Convert examples to features (VALIDATION).')
    dev_encoded_dataset = convert_examples_to_features(dev_dataset, max_seq_length=args.seq_length, tokenizer=tokenizer, srl_predictor=None)

    logger.info('Loading the model.')

    if args.mapping == 'dream':
        tag_tokenizer = TagTokenizer(vocab=['verb', 'argument', '[CLS]', '[SEP]', '[PAD]', 'location', 'temporal', 'O'])
    elif args.mapping == 'tags1':
        tag_tokenizer = TagTokenizer(vocab=['V', 'ARG1','ARG0','ARG2', 'ARG3','ARG4', '[CLS]', '[SEP]', '[PAD]', 'TMP', 'LOC', 'ARGM', 'O'])
    elif args.mapping == 'binary':
        tag_tokenizer = TagTokenizer(vocab=['verb', 'argument', '[CLS]', '[SEP]', '[PAD]', 'O'])
    else:
        tag_tokenizer = TagTokenizer()

    num_labels = 3
    vocab_size = len(tag_tokenizer.ids_to_tags) # currently 22
    logger.info('Tag vocabulary size: %d' % (vocab_size))
    logger.info(tag_tokenizer.ids_to_tags)
    tag_config = TagConfig(tag_vocab_size=vocab_size, hidden_size=10, layer_num=1, output_dim=10, dropout_prob=0.1, num_aspect=args.max_num_aspect)

    if args.aggregate:
        model = BertForSequenceClassificationTagWithAgg.from_pretrained(model_checkpoint, num_labels=num_labels, tag_config=tag_config)
    else:
        model = BertForSequenceClassificationTag.from_pretrained(model_checkpoint, num_labels = num_labels,tag_config=tag_config)

    logger.info('Create tensors.')
    # CREATE TENSOR DATASETS
    train_features = transform_tag_features(args.max_num_aspect, train_encoded_dataset, tag_tokenizer, max_seq_length=args.seq_length) #max_num_aspect=3, number of propositions, what if they were from all the evidences
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    all_start_end_idx = torch.tensor([f.orig_to_token_split_idx for f in train_features], dtype=torch.long)
    all_input_tag_ids = torch.tensor([f.input_tag_ids for f in train_features], dtype=torch.long)
    all_train_indexes = torch.tensor([f.index_id for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_end_idx, all_input_tag_ids, all_label_ids, all_train_indexes)

    eval_features = transform_tag_features(args.max_num_aspect, dev_encoded_dataset, tag_tokenizer, max_seq_length=args.seq_length)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    all_start_end_idx = torch.tensor([f.orig_to_token_split_idx for f in eval_features], dtype=torch.long)
    all_input_tag_ids = torch.tensor([f.input_tag_ids for f in eval_features], dtype=torch.long)
    all_dev_indexes = torch.tensor([f.index_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_end_idx, all_input_tag_ids, all_label_ids, all_dev_indexes)

    # DATALOADER
    batch_size = args.batch_size
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)# Create the DataLoader for our validation set.
    validation_sampler = SequentialSampler(eval_data)
    validation_dataloader = DataLoader(eval_data, sampler=validation_sampler, batch_size=batch_size)

    # PARAMETERS
    seed_val = 1995
    nb_tr_steps = 0
    tr_loss = 0
    best_epoch = 0
    best_result = 0.0
    epochs = 4
    learning_rate = 2e-5

    class ExtendedDataParallel(nn.DataParallel): # creates a dataparallel but retrieves parameters of the former model
        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

    cuda_devices = list(map(int, args.cuda_devices.split(',')))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ExtendedDataParallel(model, device_ids=cuda_devices)

    optimizer = AdamW(model.parameters(),lr=learning_rate, eps=1e-8)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    model.to(cuda_devices[0])

    # TRAINING LOOP
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    loss_values = []
    for epoch_i in range(0, epochs):
        # TRAINING
        logger.info('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        logger.info('Training...')
        t0 = time.time()
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(train_dataloader):
            if step % 500 == 0 and not step == 0: # Progress update every 500 batches.
                elapsed = format_time(time.time() - t0)
                logger.info('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, start_end_idx, input_tag_ids, label_ids, index_ids = batch
            if args.aggregate:
                loss = model(input_ids, index_ids, segment_ids, input_mask, start_end_idx, input_tag_ids, label_ids)
            else:
                loss = model(input_ids, input_tag_ids, segment_ids, input_mask, start_end_idx, label_ids)

            if len(cuda_devices) > 1:
                loss = loss.mean()

            tr_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        avg_train_loss = tr_loss / len(train_dataloader)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)
        logger.info("  Average training loss: {0:.2f}".format(avg_train_loss))
        logger.info("  Training epoch took: {:}".format(format_time(time.time() - t0)))

        # VALIDATION
        logger.info("Running Validation...")
        t0 = time.time()
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        for batch in validation_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, start_end_idx, input_tag_ids, label_ids, index_ids = batch

            with torch.no_grad():
                if args.aggregate:
                    logits = model(input_ids, index_ids, segment_ids, input_mask, start_end_idx, input_tag_ids, None)
                else:
                    logits = model(input_ids, input_tag_ids, segment_ids, input_mask, start_end_idx, None)

            # GET THE LOGITS & LABELS
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()


            # Calculate the accuracy for this batch of instances.
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            print(eval_accuracy, tmp_eval_accuracy)
            eval_accuracy += tmp_eval_accuracy # sum of accuracies is enough to know which are the best results
            nb_eval_steps += 1
            if eval_accuracy > best_result:
                best_epoch = epoch_i
                best_result = eval_accuracy
                torch.save({'epoch': epoch_i,
                            'model': model.state_dict(),
                            'best_accuracy': best_result,
                            'train_losses': tr_loss,
                           'eval_losses': eval_loss},
                           '%s/best.pth.tar' % dir_path)


        logger.info("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
        logger.info("  Validation took: {:}".format(format_time(time.time() - t0)))

        if args.vote:
            index_ids = index_ids.tolist()
            predictions = np.argmax(logits, axis=1)
            final_predictions, indexes = top_vote(list(zip(index_ids, predictions)))
            final_labels, indexes = top_vote(list(zip(index_ids, label_ids)))
            logger.info(np.sum(final_predictions == final_labels) / len(final_labels))

        torch.save({'epoch': epoch_i,
                    'model': model.state_dict(),
                    'best_accuracy': best_result,
                    'train_losses': tr_loss,
                   'eval_losses': eval_loss},
                   '%s/epoch.%d.pth.tar' % (dir_path, epoch_i))

    logger.info("best epoch: %s, result:  %s", str(best_epoch), str(best_result))

    logger.info("Training complete!")

    fout = open(dir_path + '/results.txt', 'w')
    fout.write('%d\t%lf\r\n' % (best_epoch, best_result))
    fout.close()