import json
import sqlite3
from tqdm import tqdm
from drqa.retriever import utils
ENCODING = 'utf-8'
DATABASE = 'data/fever/fever.db'

conn = sqlite3.connect(DATABASE)
cursor = conn.cursor()

'''
    Build train/dev/test set from retrieval results for BERT.
'''
def process(input, output, database):
    fin = open(input, 'rb')
    instances = []
    index = 0
    for line in fin:
        object = json.loads(line.decode(ENCODING).strip('\r\n'))
        label = ''.join(object['label'].split(' '))
        evidences = object['evidence']
        claim = object['claim']
        instances.append([index, label, claim, evidences])
        index += 1
    fin.close()
    print(index)

    with open(database) as f:
        db = json.load(f)
        counter = 0
        ids_no_evid = []

        fout = open(output, 'wb')
        for instance in tqdm(instances):
            at_least_one = False
            index, label, claim, evidences = instance
            for set_evidences in evidences:
                if len(set_evidences) == 0:
                    continue
                for evidence in set_evidences:
                    page = evidence[0]
                    n_line = evidence[1]
                    evidence_str = None

                    doc_lines = db[page]['lines']
                    # doc_lines = db[page]
                    for i in range(len(doc_lines)):
                        if i == n_line:
                            #evidence_str.append(doc_lines[i])
                            evidence_str = doc_lines[i]
                    #evidence_str = ' '.join(evidence_str)
                    confidence = 1

                    if evidence_str:
                        at_least_one = True
                        fout.write(('%s\t%s\t%s\t%s\t%s\t%d\t%s\r\n' % (label, evidence_str, claim, index, evidence[0], evidence[1], confidence)).encode(ENCODING))
                    #else:
                    #    print('Error: cant find %s %d for %s' % (url, n_line, index))
            if not at_least_one:
                ids_no_evid.append(instance[0])
                counter += 1
        print(counter)
        print(ids_no_evid)
    fout.close()


if __name__ == '__main__':
    #build_bert_train_sr_set('data/fever/train.jsonl', 'data/gear/bert-nli-train-sr-set.tsv')
    #process('data/retrieved/train.ensembles.s10.jsonl', 'data/gear/bert-nli-train-retrieve-set.tsv')
    process('data/ukp_snopes_corpus/datasets/snopes.dev.jsonl', 'data/gear/bert-snopes-dev.tsv', 'data/ukp_snopes_corpus/datasets/snopes.page.json')
    #process('data/retrieved/test.ensembles.s10.jsonl', 'data/gear/bert-nli-test-retrieve-set.tsv')
