
import argparse
import json
import csv
import random
import os

def load_fever(file):
    list_jsons=[]
    with open(file, 'r') as json_file:
        json_list = list(json_file)
    for json_str in json_list:
        result = json.loads(json_str)
        list_jsons.append(result)
    return list_jsons

def load_hover(dataset):
    data = json.load(open(dataset))
    for d in data:
        d['id'] = d.pop('uid')
    return data

def load_multifc(dataset):
    with open(dataset) as f:
        reader = csv.reader(f, skipinitialspace=True, delimiter='\t')
        header = ['id', 'label', 'claim', 'ev1', 'ev2', 'ev3', 'ev4', 'ev5']
        data = [dict(zip(header, row)) for row in reader]
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='The path to the dataset to be annotated')
    parser.add_argument('type', type=str, help='The type of dataset: MultiFC, FEVER, Hover')
    args = parser.parse_args()

    if args.type == "MultiFC":
        data = load_multifc(args.dataset)
    if args.type == "FEVER":
        data = load_fever(args.dataset)
    if args.type == "Hover":
        data = load_hover(args.dataset)
    print(data[0])

    sampling = random.choices(data, k=100)
    print(len(sampling))
    with open('Claim-Verification-FakeNews/data/annotations/'+ args.type +'.csv', 'a+', encoding='utf-8') as f:
        writer = csv.writer(f)
        if os.path.getsize('Claim-Verification-FakeNews/data/annotations/'+ args.type +'.csv') == 0:
            fieldnames = ['id', 'complexity', 'time', 'math', 'dataset']
            writer.writerow(fieldnames)
        for instance in sampling:
            print(instance['claim'])
            complexity = input('Complexity: ')
            time = input('Time: ')
            math = input('Math: ')
            writer.writerow([instance['id'], complexity, time, math, args.type])
            f.flush()

main()