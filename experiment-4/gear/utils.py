import numpy as np
import torch
import json


def feature_pooling(features, size):
    if len(features) == 0:
        features = []
        for i in range(size):
            features.append([0.0 for _ in range(768)])

    while len(features) < size:
        features.append([0.0 for _ in range(len(features[0]))])

    return np.array(features)


def load_bert_features_claim(file, size, claim_size):
    print('Loading data.')
    features, labels, claims = [], [], []
    label_to_num = {'SUPPORTS': 0, 'REFUTES': 1, 'NOTENOUGHINFO': 2}

    with open(file, 'rb') as fin:
        cnt = 0
        avg_size = avg_size_claim = []
        for line in fin:
            cnt += 1
            if cnt % 10000 == 0:
                print(cnt)
            instance = json.loads(line)
            avg_size.append(len(instance['evidences']))
            avg_size_claim.append(len(instance['claim']))
            if len(instance['evidences']) > size:
                instance['evidences'] = instance['evidences'][:size]
            if len(instance['claim'])==1:
                claim = instance['claim']
            elif len(instance['claim']) > claim_size:
                instance['claim'] = instance['claim'][:claim_size]
                claim = feature_pooling(instance['claim'], claim_size)
            feature = feature_pooling(instance['evidences'], size)
            features.append(feature)
            labels.append(label_to_num[instance['label']])
            claims.append(claim)

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    claims = torch.FloatTensor(claims)
    print('Average Claim Length:', sum(avg_size_claim) / len(avg_size_claim))
    print('Average Evidence Length:',sum(avg_size) / len(avg_size))
    return features, labels, claims


def load_bert_features_claim_test(file, size):
    features, claims = [], []

    with open(file, 'rb') as fin:
        cnt = 0
        for line in fin:
            cnt += 1
            if cnt % 10000 == 0:
                print(cnt)
            instance = json.loads(line)
            if len(instance['evidences']) > size:
                instance['evidences'] = instance['evidences'][:size]

            feature = feature_pooling(instance['evidences'], size)
            features.append(feature)
            claims.append(instance['claim'])

    features = torch.FloatTensor(features)
    claims = torch.FloatTensor(claims)
    print('Done.')
    return features, claims


def correct_prediction(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct
