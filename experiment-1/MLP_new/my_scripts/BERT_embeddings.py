
from common.features.feature_function import FeatureFunction
from common.util.array import flatten

import numpy as np
import pickle

from common.util.log_helper import LogHelper

from pytorch_pretrained_bert.tokenization import BertTokenizer
from new_features.extractor import convert_examples_to_features


class BERTFeatureFunction(FeatureFunction):


    def __init__(self,doc_db,lim_unigram=5000,naming=None,gold=True):
        super().__init__()
        self.doc_db = doc_db
        self.lim_unigram = lim_unigram
        self.naming = naming
        self.logger = LogHelper.get_logger(self.get_name())
        if gold:
            self.ename = "evidence"
        else:
            self.ename = "predicted"

    def get_name(self):
        return type(self).__name__ + (("-" + self.naming) if self.naming is not None else "")

    def inform(self,train,dev=None,test=None):
        claims = self.claims(train)
        bodies = self.bodies(train)

        if dev is not None:
            dev_claims = self.claims(dev)
            dev_bodies = self.bodies(dev)
        else:
            dev_claims = []
            dev_bodies = []

        if test is not None:
            test_claims = self.claims(test)
            test_bodies = self.bodies(test)
        else:
            test_claims = []
            test_bodies = []


    def save(self,mname):
        self.logger.info("Saving BERT features to disk")

        with open("features/{0}-bowv".format(mname), "wb+") as f:
            pickle.dump(self.bow_vectorizer, f)
        with open("features/{0}-bow".format(mname), "wb+") as f:
            pickle.dump(self.bow, f)
        with open("features/{0}-tfidf".format(mname), "wb+") as f:
            pickle.dump(self.tfidf_vectorizer, f)
        with open("features/{0}-tfreq".format(mname), "wb+") as f:
            pickle.dump(self.tfreq_vectorizer, f)


    def load(self,mname):
        self.logger.info("Loading TFIDF features from disk")

        try:
            with open("features/{0}-bowv".format(mname), "rb") as f:
                bow_vectorizer = pickle.load(f)
            with open("features/{0}-bow".format(mname), "rb") as f:
                bow = pickle.load(f)
            with open("features/{0}-tfidf".format(mname), "rb") as f:
                tfidf_vectorizer = pickle.load(f)
            with open("features/{0}-tfreq".format(mname), "rb") as f:
                tfreq_vectorizer = pickle.load(f)

            self.bow = bow
            self.bow_vectorizer = bow_vectorizer
            self.tfidf_vectorizer = tfidf_vectorizer
            self.tfreq_vectorizer = tfreq_vectorizer


        except Exception as e:
            raise e





    def lookup(self,data): #Here is where it arrives from feature_function.py
        return self.process(data)

    def process(self,data): # here is basically where the features are obtained
        claim = self.claims(data)
        evidences = self.texts(data) # it looks for the evidences in BERT_features
        claims_evidences = zip(claim, evidences)

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)
        print('Features')
        features = convert_examples_to_features(examples=claims_evidences, seq_length=128, tokenizer=tokenizer)
        #print('Evidence features')
        #features_evidence = convert_examples_to_features(examples=evidences, seq_length=60, tokenizer=tokenizer)

        # claim_bow = self.bow_vectorizer.transform(self.claims(data))
        # claim_tfs = self.tfreq_vectorizer.transform(claim_bow)
        # claim_tfidf = self.tfidf_vectorizer.transform(self.claims(data))
        #
        # body_texts = self.texts(data)
        # body_bow = self.bow_vectorizer.transform(body_texts)
        # body_tfs = self.tfreq_vectorizer.transform(body_bow)
        # body_tfidf = self.tfidf_vectorizer.transform(body_texts)
        #
        # cosines = np.array([cosine_similarity(c, b)[0] for c,b in zip(claim_tfidf,body_tfidf)])
        #
        # print(hstack([body_tfs,claim_tfs,cosines]))
        #print(hstack([features_claim, "[SEP]", features_evidence]))

        return features


    def claims(self,data):
        return [datum["claim"] for datum in data]

    def bodies(self,data): # returns the raw text in the wikipedia database, for the characters in evidence
        bodies=[]
        for id in set(flatten(self.body_ids(data))):
                b=self.doc_db.get_doc_text(id)
                if b is None:
                        bodies.append('Evidence not found')
                else:
                        bodies.append(b)
        return bodies

    def texts(self,data):
        texts = []
        counter = 0
        for instance in self.body_ids(data):
            for page in instance:
                text = self.doc_db.get_doc_text(page)
                if text is None:
                    text = ' '
                    counter += 1
                texts += ' ' + text
        print('Evidences not found: {0}'.format(counter))
        return texts


    def body_ids(self,data): #selecciona uns carácters dintre de la variable evidence que no sé d'on han sortit
        return [[d[0] for d in datum[self.ename] ] for datum in data]
