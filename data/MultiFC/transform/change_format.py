import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, help='The path to the dataset to be transformed')
parser.add_argument('--output', type=str, help='The path to the resulting dataset')
args = parser.parse_args()

with open(args.dataset, mode='r', newline='\r\n') as f:
	with open(args.output, mode='w', newline='\n') as t:
		data = f.read()
		data = data.split('\r\n')
		writer = csv.writer(t, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
		for row in data:
			row = row.split('\t')
			writer.writerow(row)