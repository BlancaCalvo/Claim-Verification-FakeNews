

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

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def read_examples_SRL(input_file, predictor):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    unique_id = line_n = 0
    num_lines = sum(1 for line in open(input_file))
    with open(input_file, "r", encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            line_n += 1
            logger.info("Parsing input with SRL: {}/{}".format(line_n, num_lines))
            if not line:
                break
            line = line.strip().split('\t')
            index = line[0]
            label = line[1]
            claim = line[2]
            evidences = line[3:]

            prediction = predictor.predict_json({'sentence': claim})
            if len(prediction['verbs']) == 0:
                examples.append(InputExample(unique_id=unique_id, text_a=claim, text_b='',
                                             label=label, index=index, is_claim=True))
                unique_id += 1
            else:
                for proposition in prediction['verbs']:
                    all_nodes = []
                    sr_parts = re.findall(r'\[[A-Z0-9]+:.*?\]', proposition['description'])
                    for part in sr_parts:
                        role, argument = part.replace('[','').replace(']','').split(': ', 1)
                        all_nodes.append(argument)
                    for pair in itertools.combinations(all_nodes, 2):
                        examples.append(InputExample(unique_id=unique_id, text_a=pair[0], text_b=pair[1],
                                                     label=label, index=index,is_claim=True))
                        unique_id += 1


            for evidence in evidences:
                evidence = re.sub(r'\.[a-zA-Z \-é0-9\(\)]*$', '', evidence) # instead of this line I should change the build_gear_input_set.py script
                try:
                    prediction = predictor.predict_json({'sentence': evidence})
                except RuntimeError:
                    print('Length issue with this evidence: ', evidence)
                    continue
                if len(prediction['verbs']) == 0:
                    continue
                for proposition in prediction['verbs']:
                    all_nodes = []
                    sr_parts = re.findall(r'\[[A-Z0-9]+:.*?\]', proposition['description'])
                    for part in sr_parts:
                        try:
                            role, argument = part.replace('[', '').replace(']', '').split(': ', 1)
                        except ValueError:
                            print('no role')
                            continue
                        all_nodes.append(argument)
                    for pair in itertools.combinations(all_nodes, 2):
                        examples.append(InputExample(unique_id=unique_id, text_a=pair[0], text_b=pair[1],
                                 label=label, index=index,is_claim=False))
                        unique_id += 1
    return examples

parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--input_file", default=None, type=str, required=True)
parser.add_argument("--output_file", default=None, type=str, required=True)
parser.add_argument("--cuda", default=-1, type=int, required=False)

args = parser.parse_args()

# load the fever dataset with evidences (the GEAR one I think)
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.11.19.tar.gz", cuda_device=args.cuda)

examples = read_examples_SRL(args.input_file, predictor)

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

model = BertModel.from_pretrained('bert-base-cased', output_hidden_states=False) #allennlp uses bert-base-uncased, I should probably reconsdier
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