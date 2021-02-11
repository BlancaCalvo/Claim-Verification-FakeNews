

from allennlp.predictors import Predictor

import re
import itertools
import logging
import argparse
import numpy as np

from extractor import InputExample
from extractor import convert_examples_to_features
from transformers import BertTokenizer, BertModel

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from tqdm import tqdm
import collections
import json

from SRL_extraction import read_examples_SRL, read_examples_SRL_1claim

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--input_file", default=None, type=str, required=True)
parser.add_argument("--output_file", default='data/graph_features/features.json', type=str, required=False)
parser.add_argument("--cuda", default=-1, type=int, required=False)
parser.add_argument('--graph_claim', dest='graph', action='store_true')
parser.add_argument('--no_graph_claim', dest='graph', action='store_false')
parser.set_defaults(graph=True)
parser.add_argument('--only_srl', dest='only_srl', action='store_true')
parser.set_defaults(only_srl=False)
parser.add_argument("--output_srl_file", default='data/graph_features/srl_file.json', type=str, required=False)
parser.add_argument("--input_srl_file", default=None, type=str, required=False)

args = parser.parse_args()

if args.input_srl_file != None:
    logger.info('Loading srl.')
    with open(args.input_srl_file, "r") as read_file:
        data = json.load(read_file)
    examples = []
    for element in data:
        examples.append(InputExample(unique_id=element['unique_id'], text_a=element['text_a'], text_b=element['text_b'],
                                     label=element['label'], index=element['index'], is_claim=element['is_claim']))
else:
    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.11.19.tar.gz", cuda_device=args.cuda)

    if args.graph:
        logger.info('Building claims as graphs.')
        examples, json_list = read_examples_SRL(args.input_file, predictor)
    else:
        logger.info('Building claims as strings.')
        examples, json_list = read_examples_SRL_1claim(args.input_file, predictor)

    with open(args.output_srl_file, 'w') as fout:
        json.dump(json_list, fout)

    if args.only_srl:
        exit()

# for each claim do SRL parsing and structure the claim so it looks like
#[CLS] became [SEP] Colin Kaepernick [SEP]
#[CLS] became [SEP] a starting quarterback [SEP]
#[CLS] became [SEP] during the 49ers 63rd season in the National Football League [SEP]
#[CLS] Colin Kaepernick [SEP] a starting quarterback [SEP]
#[CLS] Colin Kaepernick [SEP] during the 49ers 63rd season in the National Football League [SEP]
#[CLS]  a starting quarterback  [SEP] during the 49ers 63rd season in the National Football League [SEP]

# FUTURE OPTION: include roles so it can be used for fine-tuning with the new vocabulary ver arg arg-tmp arg-loc

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

features = convert_examples_to_features(
         examples=examples, seq_length=20, tokenizer=tokenizer)

local_rank = -1 #revise this later
batch_size = 512
output_file = args.output_file

if local_rank == -1 or args.cuda != -1:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cuda != -1 else "cpu")
    n_gpu = torch.cuda.device_count()
else:
    device = torch.device("cuda", local_rank)
    n_gpu = 1
#     # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
#     torch.distributed.init_process_group(backend='nccl')
logger.info("device: {} n_gpu: {} distributed training: {}".format(device, n_gpu, bool(local_rank != -1)))

model = BertModel.from_pretrained('bert-base-cased', output_hidden_states=False)
model.to(device)

if local_rank != -1:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank)
elif n_gpu > 1:
    model = torch.nn.DataParallel(model)

all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
all_segment_ids = torch.tensor([f.input_type_ids for f in features], dtype=torch.long)
all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)

# '''
# if args.local_rank == -1:
#     eval_sampler = SequentialSampler(eval_data)
# else:
#     eval_sampler = DistributedSampler(eval_data)
# '''
eval_sampler = SequentialSampler(eval_data)  # returns indices
eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

model.eval()  # turns the process into evaluation mode
sentence_embeddings = []
for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader, desc="testing"):
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        # _, pooled_output = model(input_ids, token_type_ids=None, attention_mask=input_mask, output_all_encoded_layers=False)
     #logger.info(pooled_output.shape)

     #sentence_embeddings.extend(pooled_output)
    sentence_embeddings.extend(outputs[1].detach().cpu().numpy())
all_label = [f.label for f in features]
all_index = [f.index for f in features]
all_is_claim = [f.is_claim for f in features]

instances = {}
for i in range(len(sentence_embeddings)):
    label = all_label[i]
    index = all_index[i]
    is_claim = all_is_claim[i]
    # embedding = sentence_embeddings[i].detach().cpu().numpy()
    embedding = sentence_embeddings[i]

    if index not in instances: # problema aquí
        instances[index] = {}
        instances[index]['label'] = label
        instances[index]['claim'] = []
        instances[index]['evidences'] = []
        if is_claim:
            instances[index]['claim'] = [embedding]
        else:
            instances[index]['evidences'] = [embedding]
    else:
        assert instances[index]['label'] == label
        if 'evidences' not in instances[index]:
            instances[index]['evidences'] = []
        if 'claim' not in instances[index]:
            instances[index]['claim'] = []
        if is_claim:
            instances[index]['claim'].append(embedding)
        else:
            instances[index]['evidences'].append(embedding)
    if args.graph == False:
        assert len(instances[index]['claim']) == 1
    else:
        assert len(instances[index]['claim']) > 0

fout = open(output_file, 'w')
logger.info("Saving.")
for instance in instances.items():
    #print(instance)
    output_json = collections.OrderedDict()
    output_json['index'] = instance[0]
    output_json['label'] = instance[1]['label']
    claims = []
    for claim in instance[1]['claim']:
        item = [round(x.item(), 6) for x in claim]
        claims.append(item)
    output_json['claim'] = claims
    evidences = []
    for evidence in instance[1]['evidences']:
        item = [round(x.item(), 6) for x in evidence]
        evidences.append(item)
    output_json['evidences'] = evidences
    fout.write(json.dumps(output_json) + '\r\n')
fout.close()