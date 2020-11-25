import pandas as pd
import csv

header_list = ['id', 'claim', 'original_label', 'url', 'reason_label', 'categories', 'speaker', 'factchecker', 'tags', 'article_title', 'publication_date', 'claim_date', 'entities']


data = pd.read_csv('data/MultiFC/train_trial.tsv', sep='\t', names=header_list)
print(len(data))


true_labels = ['conclusion: accurate', 'truth!', 'correct', 'true',  'determination: true', 'factscan score: true', 'verdict: true', 'fact', 'accurate' ]

false_labels = ['virus!', '3 pinnochios', 'facebook scams', 'scam!', 'we rate this claim false',  'rating: false', 'fake', 'false' , '4 pinnochios', 'determination: false', 'fake news', 'verdict: false', 'scam', 'conclusion: false', 'factscan score: false', 'incorrect',  'fiction', 'fiction!']

not_enough_info = ['no evidence', 'unproven!', 'conclusion: unclear', 'unobservable' , 'unproven', 'verdict: unsubstantiated', 'unsupported', 'conclusion: unclear', 'unsubstantiated messages']

def group_labels(original_label):
	if original_label in true_labels:
		return 'SUPPORTS', 'VERIFIABLE'
	elif original_label in false_labels:
		return 'REFUTES', 'VERIFIABLE'
	elif original_label in not_enough_info:
		return 'NOTENOUGHINFO', 'NOT VERIFIABLE'
	else:
		return 'other', 'NOT VERIFIABLE'

data['label'] = data.apply (lambda row: group_labels(row['original_label'])[0], axis=1)

data['verifiable']= data.apply (lambda row: group_labels(row['original_label'])[1], axis=1)

data = data[data['label']!='other']

print(data.groupby('label').count()['id'])

def import_snippets(id_claim):
	evidences = []
	header_snippets = ['rank_position', 'title', 'snippet', 'snippet_url']
	try:
		snippets = pd.read_csv('data/MultiFC/snippets/' + id_claim, sep='\t', names=header_snippets)
		found = snippets['snippet'].to_list()
		for f in found:
			f=f.replace('\r\n', ' ')
			f = f.replace('\n', ' ')
			f = f.replace('\r', ' ')
			f = f.replace('\t', '')
			evidences.append(str(f))
	except:
		evidences = ['','','','','']
	return evidences[:5]

# create tsv with no column names with the order
# id	label	claim	evidence 	evidence	evidence	evidence	evidence

for index, row in data.iterrows():
	evidences = import_snippets(row['id'])
	for e, evid in enumerate(evidences):
		#row['evidence' + str(e)] = evid
		data.at[index, 'evidence' + str(e)] = evid

new_order = [0, 13, 1, 15, 16, 17, 18, 19]
data = data[data.columns[new_order]]

print(data.head())
print(data.groupby('label').count()['id'])

#data = data[~data.claim.str.contains("XYZ")]
#data = data.values.tolist()

#with open('output.tsv', 'wt') as out_file:
#	tsv_writer = csv.writer(out_file, delimiter='\t')
#	for line in data:
#		print(line[1])
#		tsv_writer.writerow(line)

#data = data.replace('http.*\r [0-9].*[\r [0-9].*]*\"', '\"', regex=True)

with open('data/MultiFC/train_data.tsv', mode='w', newline='\n') as f:
	data.to_csv(f, sep='\t', header=False, index=False, line_terminator='\n', encoding='utf-8')

#data.to_csv('data/MultiFC/train_data.tsv', sep='\t', header=False, line_terminator='\n') #, quotechar='"', quoting=csv.QUOTE_NONNUMERIC
