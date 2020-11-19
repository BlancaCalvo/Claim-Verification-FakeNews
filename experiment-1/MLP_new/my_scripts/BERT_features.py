from common.util.random import SimpleRandom
from my_scripts.BERT_embeddings import BERTFeatureFunction
import os

class BERTfeatures(BERTFeatureFunction): #first thing it does is going to TermFrequency?
    def __init__(self,doc_db,lim_unigram=5000,naming=None):
        super().__init__(doc_db,lim_unigram,naming=naming)
        self.ename = "evidence"

    def texts(self,data): #then here it gets the lines of the evidences and concatenates them together with the claim
        return [" ".join(set(instance)) for instance in self.body_lines(data)]

    def body_lines(self,data):
        datums=[]
        counter= 0
        for datum in data:
            ds = []
            for d in datum[self.ename]:
                if d[0] is None or d[0]=='Evidence not found' or d[0]=='empty':
                    result = ' '
                    counter += 1
                else:
                    result = self.get_doc_line(d[0], d[1])
                ds.append(result)
            datums.append(ds)
        print('Non-found evidences: {0}'.format(counter))
        return [[self.get_doc_line(d[0],d[1]) for d in datum[self.ename] ] for datum in data]

    def get_doc_line(self,doc,line):
        lines = self.doc_db.get_doc_lines(doc)

        if os.getenv("PERMISSIVE_EVIDENCE","n").lower() in ["y","yes","true","t","1"]:
            if lines is None:
                return ""

        if line > -1:
            try:
                return lines.split("\n")[line].split("\t")[1]
            except:
                return ' '
        else:
            non_empty_lines = [line.split("\t")[1] for line in lines.split("\n") if len(line.split("\t"))>1 and len(line.split("\t")[1].strip())]
            return non_empty_lines[SimpleRandom.get_instance().next_rand(0,len(non_empty_lines)-1)]