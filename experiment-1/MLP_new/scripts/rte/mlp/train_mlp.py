import argparse

import torch
import os

from common.dataset.data_set import DataSet
from common.dataset.reader import JSONLineReader
from common.features.feature_function import Features
from common.training.early_stopping import EarlyStopping
from common.training.options import gpu
from common.training.run import train, print_evaluation
from common.util.log_helper import LogHelper
from common.util.random import SimpleRandom
from retrieval.fever_doc_db import FeverDocDB
from rte.riedel.data import FEVERGoldFormatter, FEVERLabelSchema
from rte.riedel.fever_features import TermFrequencyFeatureFunction
from rte.riedel.model import SimpleMLP
from rte.riedel.sent_features import SentenceLevelTermFrequencyFeatureFunction
from my_scripts.BERT_features import BERTfeatures



def model_exists(mname):
    return os.path.exists(os.path.join("models","{0}.model".format(mname)))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    SimpleRandom.set_seeds()

    LogHelper.setup()
    logger = LogHelper.get_logger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument('db', type=str, help='db file path')
    parser.add_argument('train', type=str, help='train file path')
    parser.add_argument('dev', type=str, help='dev file path')
    parser.add_argument('--test', required=False ,type=str, default=None ,help="test file path")
    parser.add_argument("--model", type=str, help="model name")
    parser.add_argument("--sentence",type=str2bool, default=False)
    parser.add_argument("--filtering",type=str, default=None)
    parser.add_argument("--features", type=str, default='BERT', help='TFIDF or BERT')
    args = parser.parse_args()

    if not os.path.exists("models"):
        os.mkdir("models")

    logger.info("Loading DB {0}".format(args.db))
    db = FeverDocDB(args.db)

    mname = args.model
    logger.info("Model name is {0}".format(mname))

    logger.info("Model loading data")
    jlr = JSONLineReader()
    formatter = FEVERGoldFormatter(None, FEVERLabelSchema())

    train_ds = DataSet(file=args.train, reader=jlr, formatter=formatter)
    dev_ds = DataSet(file=args.dev, reader=jlr, formatter=formatter)

    train_ds.read()
    dev_ds.read()

    test_ds = None
    if args.test is not None:
        test_ds = DataSet(file=args.test, reader=jlr, formatter=formatter)
        test_ds.read()

    ffns = []
    logger.info("Loading BERT features")
    if args.features == 'BERT':
        ffns.append(BERTfeatures(db, naming=mname))
    elif args.features == 'TFIDF':
        ffns.append(SentenceLevelTermFrequencyFeatureFunction(db, naming=mname))
    f = Features(mname, ffns)
    logger.info("Let's start loads!")
    train_feats, dev_feats, test_feats = f.load(train_ds, dev_ds, test_ds)


    print(train_feats[0][0])
    print(len(train_feats[0][0]))
    print(train_ds.data[0])

    input_shape = len(train_feats[0][0])

    model = SimpleMLP(input_shape,100,3)

    if gpu():
        model.cuda()

    #if model_exists(mname) and os.getenv("TRAIN","").lower() not in ["y","1","t","yes"]:
    #    model.load_state_dict(torch.load("models/{0}.model".format(mname)))
    #else:
    train(model, train_feats, 500, 1e-2, 90,dev_feats,early_stopping=EarlyStopping(mname))
    torch.save(model.state_dict(), "models/{0}.model".format(mname))


    print_evaluation(model, dev_feats, FEVERLabelSchema())

    if args.test is not None:
        print_evaluation(model, test_feats, FEVERLabelSchema())
