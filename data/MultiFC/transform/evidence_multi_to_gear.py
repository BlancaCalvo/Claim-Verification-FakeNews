import pandas as pd
import csv
import argparse
import re
import nltk.data
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

def group_labels(original_label):
	true_labels = ['conclusion: accurate', 'truth!', 'correct', 'true', 'determination: true', 'factscan score: true',
				   'verdict: true', 'fact', 'accurate']

	false_labels = ['virus!', '3 pinnochios', 'facebook scams', 'scam!', 'we rate this claim false', 'rating: false',
					'fake', 'false', '4 pinnochios', 'determination: false', 'fake news', 'verdict: false', 'scam',
					'conclusion: false', 'factscan score: false', 'incorrect', 'fiction', 'fiction!']

	not_enough_info = ['no evidence', 'unproven!', 'conclusion: unclear', 'unobservable', 'unproven',
					   'verdict: unsubstantiated', 'unsupported', 'conclusion: unclear', 'unsubstantiated messages']

	if original_label in true_labels:
		return 'SUPPORTS', 'VERIFIABLE'
	elif original_label in false_labels:
		return 'REFUTES', 'VERIFIABLE'
	elif original_label in not_enough_info:
		return 'NOTENOUGHINFO', 'NOT VERIFIABLE'
	else:
		return 'other', 'NOT VERIFIABLE'

def import_snippets(id_claim):
	evidences = []
	header_snippets = ['rank_position', 'title', 'snippet', 'snippet_url']
	try:
		snippets = pd.read_csv('data/MultiFC/new_snippets/' + id_claim, sep='\t', names=header_snippets)
		found = snippets['snippet'].to_list()
		for f in found:
			f = str(f)
			f = f.replace('...', ' ')
			f = re.sub(r"(^[A-Z][a-z]{2} [0-9]{1,2}, [0-9]{4})", ' ', f)
			if f == 'nan':
				f=''
			evidences.append(f)
	except:
		evidences = ['','','','','']
	return evidences[:5]

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('dataset', type=str, help='The path to the dataset to be transformed')
	parser.add_argument('--output', type=str, help='The path to the resulting dataset')
	args = parser.parse_args()

	header_list = ['id', 'claim', 'original_label', 'url', 'reason_label', 'categories', 'speaker', 'factchecker',
				   'tags', 'article_title', 'publication_date', 'claim_date', 'entities']

	data = pd.read_csv(args.dataset, sep='\t', names=header_list)
	print(len(data))

	data = data.dropna(subset=['id'])
	data['domain'] = data.apply(lambda row: row.id[:4], axis=1)
	#print(data.groupby(['domain', 'original_label']).count()['id'])

	data['label'] = data.apply (lambda row: group_labels(row['original_label'])[0], axis=1)
	data['verifiable']= data.apply (lambda row: group_labels(row['original_label'])[1], axis=1)
	data = data[data['label']!='other']

	not_enough = data[data['label']=='NOTENOUGHINFO']
	data=data.dropna(subset=['reason_label'])
	data = data[data.reason_label != 'None']
	data = data[data.reason_label != 'No.']
	data = pd.concat([data, not_enough])

	print(data.groupby(['label']).count()['id'])
	#print(data.groupby(['domain', 'label']).count()['id'])

	# create tsv with no column names with the order
	# id	label	claim	evidence 	evidence	evidence	evidence	evidence

	for index, row in data.iterrows():
		if row['label'] == "NOTENOUGHINFO":
			evidences = import_snippets(row['id'])
		else:
			evidences = sent_detector.tokenize(row['reason_label'].strip())
		for e, evid in enumerate(evidences[:5]):
			data.at[index, 'evidence' + str(e)] = evid

	new_order = [0, 14, 1, 16, 17, 18, 19, 20]
	data = data[data.columns[new_order]]

	if args.output:
		with open(args.output, mode='w', newline='\n') as f:
			data.to_csv(f, sep='\t', header=False, index=False, line_terminator='\n', encoding='utf-8')

if __name__ == "__main__":
    main()