"""Script to serialize the saliency with gradient approaches and occlusion."""
import argparse
import json
import os
import random
#from argparse import Namespace
from collections import defaultdict
from functools import partial

import numpy as np
import torch
from captum.attr import DeepLift, GuidedBackprop, InputXGradient, Occlusion, \
    Saliency, configure_interpretable_embedding_layer, \
    remove_interpretable_embedding_layer
from pypapi import events, papi_high as high
#from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertForSequenceClassification, \
    BertTokenizer

from base_bert.extractor import convert_examples_to_features, InputExample
#from base_bert.fever_bert import read_examples
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

#from models.data_loader import get_collate_fn, get_dataset
#from models.model_builder import CNN_MODEL, LSTM_MODEL

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
            #new_evidences = []
            #for evidence in evidences:
            #    evidence = re.sub(r'\.[a-zA-Z \-é0-9\(\)]*$', '', evidence)  # run this if fever_bert_srl gives bad score
            #    new_evidences.append(evidence)
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

def summarize_attributions(attributions, type='mean', model=None, tokens=None):
    if type == 'none':
        return attributions
    elif type == 'dot':
        embeddings = get_model_embedding_emb(model)(tokens)
        attributions = torch.einsum('bwd, bwd->bw', attributions, embeddings)
    elif type == 'mean':
        attributions = attributions.mean(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
    elif type == 'l2':
        attributions = attributions.norm(p=1, dim=-1).squeeze(0)
    return attributions


class BertModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(BertModelWrapper, self).__init__()
        self.model = model

    def forward(self, input, attention_mask, labels):
        return self.model(input, attention_mask=attention_mask)[0]


def get_model_embedding_emb(model):
    if args.model == 'trans':
        return model.bert.embeddings.embedding.word_embeddings
    else:
        return model.embedding.embedding


def generate_saliency(model_path, saliency_path, saliency, aggregation):
    #checkpoint = torch.load(model_path,
    #                        map_location=lambda storage, loc: storage)
    #print(checkpoint['model'].keys())
    #model_args = Namespace(**checkpoint['args'])
    labels = 3
    batch_size = 16

    transformer_config = BertConfig.from_pretrained('bert-base-uncased',
                                                        num_labels=labels)
    model_cp = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', config=transformer_config).to(
            device)
    checkpoint = torch.load(model_path+'/best.pth.tar',
                                map_location=lambda storage, loc: storage)
    model_cp.load_state_dict(checkpoint['model'])
    model = BertModelWrapper(model_cp)

    model.train()

    if saliency == 'deeplift':
        ablator = DeepLift(model)
    elif saliency == 'guided':
        ablator = GuidedBackprop(model)
    elif saliency == 'sal':
        ablator = Saliency(model)
    elif saliency == 'inputx':
        ablator = InputXGradient(model)
    elif saliency == 'occlusion':
        ablator = Occlusion(model)

    #coll_call = get_collate_fn(dataset=args.dataset, model=args.model)

    #return_attention_masks = args.model == 'trans'

    #collate_fn = partial(coll_call, tokenizer=tokenizer, device=device,
    #                     return_attention_masks=return_attention_masks,
    #                     pad_to_max_length=pad_to_max)
    #test = get_dataset(path=args.dataset_dir, mode=args.split,
    #                   dataset=args.dataset)

    #test_dl = DataLoader(batch_size=batch_size, dataset=test, shuffle=False, collate_fn=collate_fn)

    test_dataset = read_examples(args.dataset_dir)

    test_encoded_dataset = convert_examples_to_features(examples=test_dataset, seq_length=250, tokenizer=tokenizer)

    # CREATE TENSORS
    test_inputs = torch.tensor([f.input_ids for f in test_encoded_dataset], dtype=torch.long)
    test_labels = torch.tensor([f.label for f in test_encoded_dataset], dtype=torch.long)
    test_masks = torch.tensor([f.input_mask for f in test_encoded_dataset], dtype=torch.long)

    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = RandomSampler(test_data)
    test_dl = DataLoader(test_data, sampler=test_sampler,
                                  batch_size=batch_size)

    # PREDICTIONS
    predictions_path = model_path + '/example-results.tsv'
    #predictions_path = None
    if not os.path.exists(predictions_path):
        predictions = defaultdict(lambda: [])
        for batch in tqdm(test_dl, desc='Running test prediction... '):
            if args.model == 'trans':
                logits = model(batch[0], attention_mask=batch[1],labels=batch[2].long())
            else:
                logits = model(batch[0])
            logits = logits.detach().cpu().numpy().tolist()
            predicted = np.argmax(np.array(logits), axis=-1)
            predictions['class'] += predicted.tolist()
            predictions['logits'] += logits

        with open(predictions_path, 'w') as out:
            json.dump(predictions, out)

    # COMPUTE SALIENCY
    if saliency != 'occlusion':
        embedding_layer_name = 'model.bert.embeddings' if args.model == \
                                                          'trans' else \
            'embedding'
        interpretable_embedding = configure_interpretable_embedding_layer(model,
                                                                          embedding_layer_name)

    class_attr_list = defaultdict(lambda: [])
    token_ids = []
    saliency_flops = []

    for batch in tqdm(test_dl, desc='Running Saliency Generation...'):
        if args.model == 'cnn':
            additional = None
        elif args.model == 'trans':
            additional = (batch[1], batch[2])
        else:
            additional = batch[-1]

        token_ids += batch[0].detach().cpu().numpy().tolist()
        if saliency != 'occlusion':
            input_embeddings = interpretable_embedding.indices_to_embeddings(
                batch[0])

        #if not args.no_time:
        #    high.start_counters([events.PAPI_FP_OPS, ])
        #for cls_ in range(checkpoint['args']['labels']):
        for cls_ in range(labels):
            if saliency == 'occlusion':
                attributions = ablator.attribute(batch[0],
                                                 sliding_window_shapes=(
                                                 args.sw,), target=cls_,
                                                 additional_forward_args=additional)
            else:
                attributions = ablator.attribute(input_embeddings, target=cls_,
                                                 additional_forward_args=additional)

            attributions = summarize_attributions(attributions,
                                                  type=aggregation, model=model,
                                                  tokens=batch[
                                                      0]).detach().cpu(

            ).numpy().tolist()
            class_attr_list[cls_] += [[_li for _li in _l] for _l in
                                      attributions]

        #if not args.no_time:
        #    saliency_flops.append(sum(high.stop_counters()) / batch[0].shape[0])

    if saliency != 'occlusion':
        remove_interpretable_embedding_layer(model, interpretable_embedding)

    # SERIALIZE
    print('Serializing...', flush=True)
    with open(saliency_path, 'w') as out:
        print(saliency_path)
        for instance_i, _ in enumerate(test_dataset):
            print(instance_i)
            saliencies = []
            for token_i, token_id in enumerate(token_ids[instance_i]):
                token_sal = {'token': tokenizer.ids_to_tokens[token_id]}
                for cls_ in range(labels):
                    token_sal[int(cls_)] = class_attr_list[cls_][instance_i][
                        token_i]
                saliencies.append(token_sal)

            out.write(json.dumps({'tokens': saliencies}) + '\n')
            out.flush()

    #return saliency_flops


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Which dataset", default='snli',
                        type=str, choices=['snli', 'imdb', 'tweet','fever'])
    parser.add_argument("--dataset_dir",
                        help="Path to the direcory with the datasets",
                        default='data/e-SNLI/dataset/',
                        type=str)
    parser.add_argument("--split", help="Which split of the dataset",
                        default='test', type=str,
                        choices=['train', 'test'])
    parser.add_argument("--no_time",
                        help="Whether to output the time for generation in "
                             "flop",
                        action='store_true',
                        default=False)
    parser.add_argument("--model", help="Which model", default='cnn',
                        choices=['cnn', 'lstm', 'trans'], type=str)
    parser.add_argument("--models_dir",
                        help="Path where the models can be found, "
                             "with a common prefix, without _1",
                        default='snli_bert', type=str)
    parser.add_argument("--gpu", help="Flag for running on gpu",
                        action='store_true', default=False)
    parser.add_argument("--seed", help="Random seed", type=int, default=73)
    parser.add_argument("--output_dir",
                        help="Path where the saliencies will be serialized",
                        default='saliency_snli',
                        type=str)
    parser.add_argument("--sw", help="Sliding window", type=int, default=1)
    parser.add_argument("--saliency", help="Saliency type", nargs='+')
    parser.add_argument("--batch_size",
                        help="Batch size for explanation generation", type=int,
                        default=None)

    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)

    device = torch.device("cuda") if args.gpu else torch.device("cpu")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print(args, flush=True)

    for saliency in args.saliency:
        print('Running Saliency ', saliency, flush=True)

        if saliency in ['guided', 'sal', 'inputx', 'deeplift']:
            aggregations = ['l2', 'mean']
        else:  # occlusion
            aggregations = ['none']

        for aggregation in aggregations:
            flops = []
            print('Running aggregation ', aggregation, flush=True)

            #base_model_name = models_dir.split('/')[-1]
            path_out = args.models_dir + 'saliency_scores/'
            if not os.path.exists(path_out):
                os.mkdir(path_out)

            for run in range(1, 4):
                generate_saliency(
                    args.models_dir,
                    os.path.join(path_out,f'scores_{saliency}_{aggregation}_{run}'),
                    saliency,
                    aggregation)

                #flops.append(np.average(curr_flops))

            #print('FLOPS', np.average(flops), np.std(flops), flush=True)
            #print()
            #print()
