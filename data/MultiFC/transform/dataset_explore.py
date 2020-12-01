
import pandas as pd
import csv
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, help='The path to the dataset to be transformed')
parser.add_argument('--output', type=str, help='The path to the resulting dataset')
args = parser.parse_args()

header_list = ['id', 'claim', 'original_label', 'url', 'reason_label', 'categories', 'speaker', 'factchecker', 'tags', 'article_title', 'publication_date', 'claim_date', 'entities']

data = pd.read_csv(args.dataset, sep='\t', names=header_list)

print(len(data))
data=data.dropna(subset=['reason_label'])
data = data[data.reason_label != 'None']
data = data[data.reason_label != 'No.']
print(len(set(data.reason_label)))
print(len(data))

# list_reasons = []
# for reason in data.reason_label:
#     if reason not in list_reasons:
#         list_reasons.append(reason)
#     else:
#         print(reason)

if args.output:
	with open(args.output, mode='w', newline='\n') as f:
		data.to_csv(f, sep='\t', header=False, index=False, line_terminator='\n', encoding='utf-8')

