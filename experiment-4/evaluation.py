import numpy as np
#from fever.scorer import fever_score
import json
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import re
import argparse
from collections import Counter

ENCODING = 'utf-8'

def get_predicted_label(items):
    labels = ['SUPPORTS', 'REFUTES', 'NOTENOUGHINFO']
    #print(labels[np.argmax(np.array(items))])
    return labels[np.argmax(np.array(items))]

def indices(l, val):
    retval = []
    last = 0
    while val in l[last:]:
        i = l[last:].index(val)
        retval.append(last + i)
        last += i + 1
    return retval

def class_accuracy(l, y_pred, y_true, classes):
    index = indices(l,classes)
    y_pred, y_true = np.array(y_pred)[index], np.array(y_true)[index]
    macro = f1_score(y_pred, y_true, average='macro')
    micro = f1_score(y_pred, y_true, average='micro')
    return micro, macro, len(y_pred)

def dev_scorer(truth_file, output_file):
    fin = open(truth_file, 'rb')

    truth_list = []
    domain = []
    for claim_num, line in enumerate(fin):
        claim_id, label, claim = line.decode(ENCODING).strip('\r\n').split('\t')[:3]
        domain.append(re.sub('-[0-9]{5}$', '', claim_id))
        truth_list.append(label)
    fin.close()

    #print(truth_list)

    fin = open(output_file, 'rb')
    lines = fin.readlines()
    answers = []
    for i in range(len(lines)):
        arr = lines[i].decode(ENCODING).strip('\r\n').split('\t')
        result = ([float(arr[0]), float(arr[1]), float(arr[2])])
        answers.append(get_predicted_label(result))
    fin.close()

    # macro_list = micro_list = 0
    # print('Micro_F1', 'Macro_F1', 'Num_Instances')
    # for d in set(domain):
    #     micro, macro, instances = class_accuracy(domain, truth_list, answers, d)
    #     print(d,micro,macro, instances)
    #     macro_list+=macro
    #     micro_list+=micro
    # print('Average', micro_list / len(set(domain)),macro_list/len(set(domain)))

    print(Counter(truth_list))

    print(accuracy_score(truth_list, answers))
    print(confusion_matrix(truth_list, answers, labels=list(set(truth_list))))
    print(f1_score(truth_list, answers, average='macro'))
    print(f1_score(truth_list, answers, average='micro'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=314, help='Random seed.')
    parser.add_argument("--pool", type=str, default="att", help='Aggregating method: top, max, mean, concat, att, sum')
    parser.add_argument("--layer", type=int, default=1, help='Graph Layer.')
    parser.add_argument("--evi_nodes", type=int, default=20, help='Evidence num.')
    parser.add_argument("--claim_nodes", type=int, default=1, help='Evidence num.')

    parser.add_argument("--dev_data", default='data/gear/gear-dev-set-0_001.tsv', type=str, required=True)
    parser.add_argument("--test_data", default='data/gear/test-short.tsv', type=str,
                        required=True)
    parser.add_argument("--note", default='_', type=str, help='Give it keyword for trial runs.')

    args = parser.parse_args()

    print('Dev score:')
    dev_scorer(args.dev_data,
               'experiment-4/outputs/gear-%devi-%dclaim-%dlayer-%s-%dseed-001%s/dev-results.tsv' % (args.evi_nodes, args.claim_nodes, args.layer, args.pool, args.seed, args.note))

    print('Test score:')
    dev_scorer(args.test_data,
               'experiment-4/outputs/gear-%devi-%dclaim-%dlayer-%s-%dseed-001%s/test-results.tsv' % (args.evi_nodes, args.claim_nodes, args.layer, args.pool, args.seed, args.note))