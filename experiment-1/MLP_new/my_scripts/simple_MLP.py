
import argparse
import os
import json
from common.dataset.reader import JSONLineReader
from rte.riedel.data import FEVERGoldFormatter, FEVERLabelSchema
from common.dataset.data_set import DataSet
from common.features.feature_function import Features
#from rte.riedel.sent_features import SentenceLevelTermFrequencyFeatureFunction
from my_scripts.BERT_features import BERTfeatures
from retrieval.fever_doc_db import FeverDocDB
from common.util.log_helper import LogHelper

#when importing a class, whatever is before the first def it's printed/loaded


if __name__ == "__main__":
    #SimpleRandom.set_seeds()

    LogHelper.setup()
    logger = LogHelper.get_logger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument('db', type=str, help='db file path')
    parser.add_argument('train', type=str, help='train file path')
    parser.add_argument('dev', type=str, help='dev file path')
    parser.add_argument("--model", type=str, help="model name")
    args = parser.parse_args()

    if not os.path.exists("models"):
        os.mkdir("models")
    mname = args.model

    # READ THE DATA
    logger.info("Model loading data")
    jlr = JSONLineReader()
    formatter = FEVERGoldFormatter(None, FEVERLabelSchema())
    train_ds = DataSet(file=args.train, reader=jlr, formatter=formatter)
    dev_ds = DataSet(file=args.dev, reader=jlr, formatter=formatter)
    train_ds.read()
    dev_ds.read() # these are a kind of datasets that have to be accessed like dev_ds.data

    # EXTRACT THE FEATURES
    db = FeverDocDB(args.db)
    ffns = []
    logger.info("Loading BERTfeatures")
    ffns.append(BERTfeatures(db, naming=mname))
    f = Features(mname, ffns)
    logger.info("Let's start loads!")
    train_feats, dev_feats, test_feats = f.load(train_ds, dev_ds)
    print(dev_feats)

    #train = load_data(args.train)
    #dev = load_data(args.dev)
