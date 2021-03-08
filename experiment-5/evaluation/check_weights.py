
import argparse
import numpy as np
import torch
import json

parser = argparse.ArgumentParser()
#
parser.add_argument("--dev_features", default='data/srl_features/dev_srl_all.json', type=str, required=False)

parser.add_argument("--concat", action='store_true', help="Set this flag if you want to concat evidences.")
parser.add_argument("--aggregate", action='store_true', help="Set this flag if you want to aggregate the evidences.") #does not work yet
parser.add_argument("--vote", action='store_true', help="Set this flag if you want to make voting system with evidences.")
parser.add_argument("--max_num_aspect", default=3, type=int, required=False)

parser.add_argument("--seq_length", default=300, type=int, required=False)
parser.add_argument("--batch_size", default=16, type=int, required=False)
#parser.add_argument("--cuda_devices", default='-1', type=str, required=False)

args = parser.parse_args()

base_dir = 'experiment-5/outputs/sembert-vote_%s-concat_%s-agg_%s-%dbatch_size-%dseq_length-%dn_aspect/' % (str(args.vote), str(args.concat), str(args.aggregate), args.batch_size, args.seq_length, args.max_num_aspect)

checkpoint = torch.load(base_dir + 'best.pth.tar')
#print(type(checkpoint['model'])) # should be dict
save_weights = dict()
#for i in checkpoint['model'].keys():
#    print(i, checkpoint['model'][i].shape)
save_weights['module.tag_model.embed.tag_embeddings.weight'] = checkpoint['model']['module.tag_model.embed.tag_embeddings.weight'].squeeze().tolist()
save_weights['module.dense.weight'] = checkpoint['model']['module.dense.weight'].squeeze().tolist()
save_weights['module.pool.weight'] = checkpoint['model']['module.pool.weight'].squeeze().tolist()

with open(base_dir+'weights.json', 'w') as fp:
    json.dump(save_weights, fp)