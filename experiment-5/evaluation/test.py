import random, os
import argparse
import numpy as np
import torch
from fever_bert_srl import read_srl_examples, read_srl_examples_concat
from transformers.models.bert.tokenization_bert import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from tagged_features import InputExample, convert_examples_to_features, transform_tag_features
from tag_model.tag_tokenization import TagTokenizer
from tag_model.modeling import TagConfig
from sembert.modeling import BertForSequenceClassificationTag, BertForSequenceClassificationTagWithAgg
from allennlp.predictors import Predictor

#from utils import load_bert_features_claim_test
#from models import GEAR

parser = argparse.ArgumentParser()
#
parser.add_argument("--dev_features", default='data/srl_features/dev_srl_all.json', type=str, required=False)

#parser.add_argument("--concat", action='store_true', help="Set this flag if you want to concat evidences.")
parser.add_argument("--aggregate", action='store_true', help="Set this flag if you want to aggregate the evidences.") #does not work yet
parser.add_argument("--no_srl", action='store_true', help="Set this flag if the given dataset does not have srl.")
parser.add_argument("--max_num_aspect", default=3, type=int, required=False)
parser.add_argument("--mapping", default=None, type=str, required=False)
parser.add_argument("--seq_length", default=300, type=int, required=False)
parser.add_argument("--batch_size", default=16, type=int, required=False)
#parser.add_argument("--cuda_devices", default='-1', type=str, required=False)

args = parser.parse_args()

if args.no_srl:
    print('Here import examples and srl predictor')
    #dev_dataset = read_examples() look at original sembert training script!
    srl_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.11.19.tar.gz")
else:
    dev_dataset = read_srl_examples_concat(args.dev_features, args.mapping)
    srl_predictor = None



model_checkpoint = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_checkpoint, do_lower_case=True)

#logger.info('Convert examples to features (VALIDATION).')
dev_encoded_dataset = convert_examples_to_features(dev_dataset, max_seq_length=args.seq_length, tokenizer=tokenizer, srl_predictor=srl_predictor)

if args.mapping == 'dream':
    tag_tokenizer = TagTokenizer(vocab=['verb', 'argument', '[CLS]', '[SEP]', '[PAD]', 'location', 'temporal', 'O'])
elif args.mapping == 'tags1':
    tag_tokenizer = TagTokenizer(
        vocab=['V', 'ARG1', 'ARG0', 'ARG2', 'ARG3', 'ARG4', '[CLS]', '[SEP]', '[PAD]', 'TMP', 'LOC', 'ARGM', 'O'])
else:
    tag_tokenizer = TagTokenizer()

# dataloader
batch_size = args.batch_size
eval_features = transform_tag_features(args.max_num_aspect, dev_encoded_dataset, tag_tokenizer, max_seq_length=args.seq_length)
all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
all_start_end_idx = torch.tensor([f.orig_to_token_split_idx for f in eval_features], dtype=torch.long)
all_input_tag_ids = torch.tensor([f.input_tag_ids for f in eval_features], dtype=torch.long)
all_dev_indexes = torch.tensor([f.index_id for f in eval_features], dtype=torch.long)
eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_end_idx, all_input_tag_ids, all_label_ids, all_dev_indexes)
validation_sampler = SequentialSampler(eval_data)
dev_dataloader = DataLoader(eval_data, sampler=validation_sampler, batch_size=batch_size)


num_labels = 3
vocab_size = len(tag_tokenizer.ids_to_tags)
tag_config = TagConfig(tag_vocab_size=vocab_size, hidden_size=10, layer_num=1, output_dim=10, dropout_prob=0.1, num_aspect=args.max_num_aspect)

if args.aggregate:
    model = BertForSequenceClassificationTagWithAgg.from_pretrained(model_checkpoint, num_labels=num_labels, tag_config=tag_config)
else:
    model = BertForSequenceClassificationTag.from_pretrained(model_checkpoint, num_labels = num_labels,tag_config=tag_config)

seeds = [1995]

device = "cuda:0"
model.to(device)

for seed in seeds:
    base_dir = 'experiment-5/outputs/oie_sembert-concat_True-agg_%s-%dbatch_size-%dseq_length-%dn_aspect-%s/' % (str(args.aggregate), args.batch_size, args.seq_length, args.max_num_aspect, str(args.mapping))
    #base_dir = 'experiment-5/outputs/bert_base/'
    if not os.path.exists(base_dir):
        print('%s results do not exist!' % base_dir)
        continue


    # FOR SOME REASON KEYS OF THE DICT CONTAIN MODULE., CHECK WHY
    checkpoint = torch.load(base_dir + 'best.pth.tar')
    #print(type(checkpoint['model'])) # should be dict
    new_checkpoint = checkpoint
    checkpoint_new_names = {}
    for i in checkpoint['model'].keys():
        checkpoint_new_names[i] = i[7:]
    new_checkpoint['model'] = dict((checkpoint_new_names[key], value) for (key, value) in checkpoint['model'].items())
    #print(new_checkpoint['model']['dense.weight'].shape)
    #print(new_checkpoint['model']['dense.bias'].shape)


    model.load_state_dict(new_checkpoint['model'])
    model.eval()

    fout = open(base_dir + 'dev-results.tsv', 'w')
    #dev_tqdm_iterator = tqdm(dev_dataloader)
    with torch.no_grad():
        for batch in dev_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, start_end_idx, input_tag_ids, label_ids, index_ids = batch

            if args.aggregate:
                logits = model(input_ids, index_ids, segment_ids, input_mask, start_end_idx, input_tag_ids, None)
            else:
                logits = model(input_ids, segment_ids, input_mask, start_end_idx, input_tag_ids, None) # RuntimeError: Input, output and indices must be on the current device

            #print(logits.shape)

            for i in range(logits.shape[0]):
                #fout.write('\t'.join(['%.4lf' % num for num in logits[i]]) + '\r\n')
                fout.write('{}\t{}\t{}\t{}\t{}\n'.format(logits[i][0], logits[i][1], logits[i][2], label_ids[i], index_ids[i]))
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
