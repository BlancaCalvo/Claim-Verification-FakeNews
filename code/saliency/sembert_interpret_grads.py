import os

import argparse
import logging
import json
from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers.models.bert.tokenization_bert import BertTokenizer

# FROM SemBERT
from tagged_features import InputExample, convert_examples_to_features, transform_tag_features
from tag_model.tag_tokenization import TagTokenizer
from tag_model.modeling import TagConfig
from sembert.modeling import BertForSequenceClassificationTag, BertForSequenceClassificationTagWithAgg
from sembert_train import read_srl_examples_concat

from captum.attr import DeepLift, GuidedBackprop, InputXGradient, Occlusion, \
    Saliency, configure_interpretable_embedding_layer, \
    remove_interpretable_embedding_layer
from collections import defaultdict

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_model_embedding_emb(model):
    return model.bert.embeddings.embedding.word_embeddings

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


def generate_saliency(model, model_path, test, data_loader, saliency_path, saliency, aggregation):
    model.train()

    labels = 3

    if saliency == 'guided':
        ablator = GuidedBackprop(model)
    elif saliency == 'sal':
        ablator = Saliency(model)
    elif saliency == 'inputx':
        ablator = InputXGradient(model)
    #elif saliency == 'occlusion':
    #    ablator = Occlusion(model)

    test_dl = data_loader

    if saliency != 'occlusion':
        #embedding_layer_name = 'model.bert.embeddings'
        interpretable_embedding = configure_interpretable_embedding_layer(model, 'bert.embeddings')
        interpretable_tag_embedding = configure_interpretable_embedding_layer(model, 'tag_model.embed')

    class_attr_list = defaultdict(lambda: [])
    class_attr_list_tags = defaultdict(lambda: [])
    token_ids = []

    for batch in tqdm(test_dl, desc='Running Saliency Generation...'):

        # SEMBERT FORWARD ARGUMENTS: forward(self, input_ids, input_tag_ids, token_type_ids=None, attention_mask=None, start_end_idx=None, labels=None, no_cuda=False, gradients_mode=False,seq_len=None):

        # DATALOADER STRUCTURE:
        # batch[0] -> input_ids
        # batch[1] -> attention_mask
        # batch[2] -> segment_ids
        # batch[3] -> start_end
        # batch[4] -> tag_ids
        # batch[5] -> labels
        # batch[6] -> indexes

        print(batch[5])

        token_ids += batch[0].detach().cpu().numpy().tolist()

        # adds the embedding dimension to the input
        input_embeddings = interpretable_embedding.indices_to_embeddings(batch[0])
        #print(input_embeddings.shape)
        tag_embeddings = interpretable_tag_embedding.indices_to_embeddings(batch[4])
        #print(tag_embeddings.shape)
        inputs = (input_embeddings, tag_embeddings)
        additional = (batch[1], batch[2], batch[3], batch[5], True, True, 250)

        for cls_ in range(labels):
            attributions = ablator.attribute(inputs, target=cls_, additional_forward_args=additional)
            #print(len(attributions)) # ATTRIBUTIONS ARE NOW DOUBLE BECAUSE WE HAVE TWO SETS OF INPUTS
            #print(attributions[1].shape) # same as input tag_ids with embedding dimension
            #print(attributions[1][0][0][0]) # non-zero
            #print(attributions[1][0][0][1]) # all zeros, as well as every other tag which is not the first one of the 250

            # get just one value for input and index (flattens again the embedding dimension)
            attributions_tokens = summarize_attributions(attributions[0],type=aggregation, model=model, tokens=batch[0]).detach().cpu().numpy().tolist()
            attributions_tags = summarize_attributions(attributions[1], type=aggregation, model=model,tokens=batch[4]).detach().cpu().numpy().tolist()

            #print(attributions_tokens[0]) # there is an attribution value for each token for each instance
            #print(attributions_tags[0]) # it returns a value for each tag, but for some reason, just the first value of the 250 is different than 0. WHY??

            class_attr_list[cls_] += [[_li for _li in _l] for _l in attributions_tokens]
            class_attr_list_tags[cls_] += [[_li for _li in _l] for _l in attributions_tags]

    remove_interpretable_embedding_layer(model, interpretable_embedding)
    remove_interpretable_embedding_layer(model, interpretable_tag_embedding)

    # SERIALIZE
    print('Serializing...', flush=True)
    with open(saliency_path, 'w') as out:
        for instance_i, _ in enumerate(test):
            saliencies = []
            prop_saliencies = []
            for token_i, token_id in enumerate(token_ids[instance_i]):
                token_sal = {'token': tokenizer.ids_to_tokens[token_id]}
                for cls_ in range(labels):
                    token_sal[int(cls_)] = class_attr_list[cls_][instance_i][token_i]
                saliencies.append(token_sal)
            for prop in range(12):
                prop_sal = {'prop': str(prop)}
                for cls_ in range(labels):
                    #print(instance_i)
                    #print(prop)
                    #print(cls_)
                    #print(class_attr_list_tags[cls_][instance_i][prop])
                    prop_sal[int(cls_)] = class_attr_list_tags[cls_][instance_i][prop][0]
                prop_saliencies.append(prop_sal)

            out.write(json.dumps({'tokens': saliencies, 'props':prop_saliencies}) + '\n')
            out.flush()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev_file", default=None, type=str, required=False)
    parser.add_argument("--batch_size", default=20, type=int, required=False)
    parser.add_argument("--seq_length", default=250, type=int, required=False)
    #parser.add_argument("--cuda_devices", default='0', type=str, required=False)
    parser.add_argument("--max_num_aspect", default=12, type=int, required=False)
    parser.add_argument("--mapping", default='tags1', type=str, required=False)
    parser.add_argument("--saliency", help="Saliency type", nargs='+')
    parser.add_argument("--model_path", default='experiment-5/outputs/f_sembert-concat_True-agg_False-20batch_size-250seq_length-12n_aspect-tags1/', type=str, required=False)
    #parser.add_argument("--path_out",default='data/saliency/sembert',type=str, required=False)
    args = parser.parse_args()

    # LOAD DATA AND MODEL AND CREATE DATALOADER
    logger.info('Loading srl.')
    dev_dataset = read_srl_examples_concat(args.dev_file, args.mapping)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    logger.info('Convert examples to features (VALIDATION).')
    dev_encoded_dataset = convert_examples_to_features(dev_dataset, max_seq_length=args.seq_length, tokenizer=tokenizer, srl_predictor=None)

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

    logger.info('Create tensors.')
    # CREATE TENSOR DATASETS
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
    validation_sampler = SequentialSampler(eval_data)
    validation_dataloader = DataLoader(eval_data, sampler=validation_sampler, batch_size=batch_size)

    # LOAD THE MODEL
    # dir_path = 'experiment-5/outputs/gold_sembert-concat_True-agg_%s-%dbatch_size-%dseq_length-%dn_aspect-%s/' % (str(args.aggregate), args.batch_size, args.seq_length, args.max_num_aspect, str(args.mapping))
    logger.info(args.model_path)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BertForSequenceClassificationTag.from_pretrained('bert-base-uncased', num_labels=3, tag_config=tag_config)#.to(device)
    checkpoint = torch.load(args.model_path + 'best.pth.tar',
                            map_location=lambda storage, loc: storage)
    new_checkpoint = checkpoint
    checkpoint_new_names = {}
    for i in checkpoint['model'].keys():
        checkpoint_new_names[i] = i[7:]
    new_checkpoint['model'] = dict((checkpoint_new_names[key], value) for (key, value) in checkpoint['model'].items())
    model.load_state_dict(checkpoint['model'])

    # COMPUTE THE GRADIENTS

    for saliency in args.saliency:
        print('Running Saliency ', saliency, flush=True)

        if saliency in ['guided', 'sal', 'inputx', 'deeplift']:
            aggregations = ['mean', 'l2']
        #else:  # occlusion
        #    aggregations = ['none']

        for aggregation in aggregations:
            print('Running aggregation ', aggregation, flush=True)

            path_out = args.model_path + 'saliency_scores/'
            if not os.path.exists(path_out):
                os.mkdir(path_out)

            print(path_out)
            for run in range(1, 4):
                generate_saliency(model, args.model_path, eval_data, validation_dataloader, os.path.join(path_out,f'scores_{saliency}_{aggregation}_{run}'), saliency, aggregation)

            print('Done')