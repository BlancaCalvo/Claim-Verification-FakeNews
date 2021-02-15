import random, os
import argparse
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from utils import load_bert_features_claim_test
from models import GEAR

parser = argparse.ArgumentParser()
parser.add_argument('--patience', type=int, default=20, help='Patience')
parser.add_argument('--seed', type=int, default=314, help='Random seed.')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')

parser.add_argument("--pool", type=str, default="att", help='Aggregating method: top, max, mean, concat, att, sum')
parser.add_argument("--layer", type=int, default=1, help='Graph Layer.')
parser.add_argument("--evi_nodes", type=int, default=5, help='Evidence num.')
parser.add_argument("--claim_nodes", type=int, default=5, help='Evidence num.')

parser.add_argument("--dev_features", default='data/graph_features/dev_features_1claim.json', type=str, required=True)
parser.add_argument("--test_features", default='data/graph_features/test_features_short.json', type=str, required=True)
parser.add_argument("--note", default='_', type=str,  help='Give it keyword for trial runs.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

dev_features, dev_claims = load_bert_features_claim_test(args.dev_features, args.evi_nodes)
test_features, test_claims = load_bert_features_claim_test(args.test_features, args.evi_nodes)

feature_num = dev_features[0].shape[1]
model = GEAR(nfeat=feature_num, nins=args.evi_nodes, nclaim=args.claim_nodes, nclass=3, nlayer=args.layer, pool=args.pool)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()

dev_features, dev_claims = Variable(dev_features), Variable(dev_claims)
test_features, test_claims = Variable(test_features), Variable(test_claims)

dev_data = TensorDataset(dev_features, dev_claims)
dev_dataloader = DataLoader(dev_data, batch_size=args.batch_size)
test_data = TensorDataset(test_features, test_claims)
test_dataloader = DataLoader(test_data, batch_size=args.batch_size)

seeds = [314]

for seed in seeds:
    base_dir = 'experiment-4/outputs/gear-%devi-%dclaim-%dlayer-%s-%dseed-001%s/' % (args.evi_nodes, args.claim_nodes, args.layer, args.pool, args.seed, args.note)
    if not os.path.exists(base_dir + 'results.txt'):
        print('%s results do not exist!' % base_dir)
        continue

    checkpoint = torch.load(base_dir + 'best.pth.tar')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    fout = open(base_dir + 'dev-results.tsv', 'w')
    dev_tqdm_iterator = tqdm(dev_dataloader)
    with torch.no_grad():
        for index, data in enumerate(dev_tqdm_iterator):
            feature_batch, claim_batch = data
            feature_batch = feature_batch.cuda()
            claim_batch = claim_batch.cuda()
            outputs = model(feature_batch, claim_batch)

            for i in range(outputs.shape[0]):
                fout.write('\t'.join(['%.4lf' % num for num in outputs[i]]) + '\r\n')
    fout.close()

    fout = open(base_dir + 'test-results.tsv', 'w')
    test_tqdm_iterator = tqdm(test_dataloader)
    with torch.no_grad():
        for index, data in enumerate(test_tqdm_iterator):
            feature_batch, claim_batch = data
            feature_batch = feature_batch.cuda()
            claim_batch = claim_batch.cuda()
            outputs = model(feature_batch, claim_batch)

            for i in range(outputs.shape[0]):
                fout.write('\t'.join(['%.4lf' % num for num in outputs[i]]) + '\r\n')
    fout.close()
