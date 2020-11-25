import csv

with open('data/MultiFC/train.tsv', mode='r', newline='\r\n') as f:
	with open('data/MultiFC/train_trial.tsv', mode='w', newline='\n') as t:
		data = f.read()
		data = data.split('\r\n')
		writer = csv.writer(t, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
		for row in data:
			row = row.split('\t')
			writer.writerow(row)