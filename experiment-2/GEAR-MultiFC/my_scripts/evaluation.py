import numpy as np
#from fever.scorer import fever_score
import json
from sklearn.metrics import accuracy_score, confusion_matrix

ENCODING = 'utf-8'

def get_predicted_label(items):
    labels = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']
    #print(labels[np.argmax(np.array(items))])
    return labels[np.argmax(np.array(items))]

def dev_scorer(truth_file, output_file):
    fin = open(truth_file, 'rb')

    truth_list = []
    for claim_num, line in enumerate(fin):
        claim_id, label, claim = line.decode(ENCODING).strip('\r\n').split('\t')[:3]
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

    #print(answers)

    print(accuracy_score(truth_list, answers))
    print(confusion_matrix(truth_list, answers))




if __name__ == '__main__':
    print('Dev score:')
    dev_scorer('data/MultiFC/dev_data.tsv',
               'experiment-2/GEAR-MultiFC/outputs/gear-5evi-1layer-att-314seed-001/dev-results.tsv')