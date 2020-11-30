import csv
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('folder', type=str, help='The path to the folder of the datasets to be transformed')
parser.add_argument('--output_dir', type=str, help='The path to the resulting datasets folder')
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

for file in os.listdir(args.folder):
	with open(args.folder+file, mode='r', newline='\r\n') as f:
		with open(args.output_dir+file, mode='w', newline='\n') as t:
			data = f.read()
			data = data.split('\r\n')
			writer = csv.writer(t, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
			for row in data:
				row = row.split('\t')
				writer.writerow(row)