import numpy as np
from fever.scorer import fever_score
import json
import argparse

ENCODING = 'utf-8'


def get_predicted_label(items):
    labels = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']
    return labels[np.argmax(np.array(items))]


def dev_scorer(truth_file, output_file, result_file, threshold):
    fin = open(truth_file, 'rb')

    truth_list = []
    for line in fin:
        label, evidence, claim, claim_num, article, article_index, confident = line.decode(ENCODING).strip('\r\n').split('\t')
        if float(confident) >= threshold:
            truth_list.append([claim_num, label, evidence, claim, article, article_index])
    fin.close()

    fin = open(output_file, 'rb')
    lines = fin.readlines()
    results = []
    for i in range(len(lines)):
        arr = lines[i].decode(ENCODING).strip('\r\n').split('\t')
        results.append([float(arr[0]), float(arr[1]), float(arr[2])])
    fin.close()

    claim2info = {}
    for item in truth_list:
        claim_num = int(item[0])
        if claim_num not in claim2info:
            claim2info[claim_num] = []
        claim2info[claim_num].append(item[1:])

    answers = []
    cnt = -1
    for i in range(0, 19998):
        answer = {}
        if i not in claim2info:
            answer = {"predicted_label": "NOT ENOUGH INFO",  "predicted_evidence": []}
            answers.append(answer)
            continue
        cnt += 1
        answer['predicted_label'] = get_predicted_label(results[cnt])
        answer["predicted_evidence"] = []
        for item in claim2info[i]:
            answer["predicted_evidence"].append([item[3], int(item[4])])

        answers.append(answer)
    true_answers = []
    fin = open(result_file, 'rb')
    lines = fin.readlines()
    for i in range(len(lines)):
        line = lines[i]
        true_answers.append(json.loads(line.decode(ENCODING).strip('\r\n')))
    fin.close()

    strict_score, label_accuracy, precision, recall, f1 = fever_score(answers, true_answers)
    print(strict_score, label_accuracy, precision, recall, f1)


def test_collector(truth_file, output_file, result_file, threshold):
    fin = open(truth_file, 'rb')

    truth_list = []
    for line in fin:
        arr = line.decode(ENCODING).strip('\r\n').split('\t')
        label = arr[0]
        evidence = arr[1]
        claim = arr[2]
        claim_num = arr[3]
        article = arr[4]
        article_index = arr[5]
        confidence = float(arr[6])

        if confidence >= threshold:
            truth_list.append([claim_num, label, evidence, claim, article, article_index])
    fin.close()

    fin = open(output_file, 'rb')
    lines = fin.readlines()
    results = []
    for i in range(len(lines)):
        arr = lines[i].decode(ENCODING).strip('\r\n').split('\t')
        results.append([float(arr[0]), float(arr[1]), float(arr[2])])
    fin.close()

    claim2info = {}
    for item in truth_list:
        claim_num = int(item[0])
        if claim_num not in claim2info:
            claim2info[claim_num] = []
        claim2info[claim_num].append(item[1:])

    claim2id = {}
    fin = open(result_file, 'rb')
    lines = fin.readlines()
    for i in range(len(lines)):
        line = lines[i]
        claim2id[i] = json.loads(line)['id']
    fin.close()

    #missing_ids = [81,404,709,896,1477,2085,2168,2210,2782,3369,3506,4408,5125,5423,5635,5676,5856,6397,6607,6722,6949,7074,7157,7325,7585,7750,8020,8473,8700,8749,9026,9270,10065,10152,10190,10701,11484,11864,12872,12900,12934,12999,13352,13620,13674,13771,14473,14548,15177,15238,15247,15531,15654,15961,16350,17185,17419,17637,18512,18889,18921,19434,19790]
    answers = []
    cnt = -1
    for i in range(0, 19998):
        answer = {}
        answer['id'] = claim2id[i]
        if i not in claim2info:
            print(answer)
            answer = {"id":answer['id'],"predicted_label": "NOT ENOUGH INFO",  "predicted_evidence": []} # this might be a problem when uploading to shared task, check why would it have no id
            print(answer)
            #del missing_ids[0]
            answers.append(answer)
            continue
        cnt += 1
        answer["predicted_label"] = get_predicted_label(results[cnt])
        answer["predicted_evidence"] = []
        for item in claim2info[i]:
            answer["predicted_evidence"].append([item[3], int(item[4])])

        answers.append(answer)

    fout = open('predictions.jsonl', 'wb')
    for answer in answers:
        fout.write(('%s\r\n' % json.dumps(answer)).encode(ENCODING))
    fout.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--predicted_labels", type=str)
    parser.add_argument("--predicted_evidence", type=str)
    parser.add_argument("--actual", type=str, required=False)
    parser.add_argument("--shared_task", type=str, required=False)
    parser.add_argument("--test", action='store_true', help="Produce test output.")
    args = parser.parse_args()

    if not args.test:
        print('Dev score:')
        dev_scorer(args.predicted_evidence, #bert-nli
               args.predicted_labels,
               args.actual, 0) #jsonl
    else:
        print('Collect test results:')
        test_collector(args.predicted_evidence,
                   args.predicted_labels,
                   args.shared_task, 0)
        print('Results can be found in predictions.jsonl')
