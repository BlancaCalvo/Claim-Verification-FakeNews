import pandas as pd

header_list = ['id', 'claim', 'original_label', 'url', 'reason_label', 'categories', 'speaker', 'factchecker', 'tags', 'article_title', 'publication_date', 'claim_date', 'entities']

data = pd.read_csv('dev.tsv', sep='\t', names=header_list)


# paper says there should be the full text of the article and of the hyperlinks, but those are not here. 

true_labels = ['conclusion: accurate', 'truth!', 'correct', 'true',  'determination: true', 'factscan score: true', 'verdict: true', 'fact', 'accurate' ]

false_labels = ['virus!', '3 pinnochios', 'facebook scams', 'scam!', 'we rate this claim false',  'rating: false', 'fake', 'false' , '4 pinnochios', 'determination: false', 'fake news', 'verdict: false', 'scam', 'conclusion: false', 'factscan score: false', 'incorrect',  'fiction', 'fiction!']

not_enough_info = ['no evidence', 'unproven!', 'conclusion: unclear', 'unobservable' , 'unproven', 'verdict: unsubstantiated', 'unsupported', 'conclusion: unclear', 'unsubstantiated messages']

def group_labels(original_label):
	if original_label in true_labels:
		return 'SUPPORTS', 'VERIFIABLE'
	elif original_label in false_labels:
		return 'REFUTES', 'VERIFIABLE'
	elif original_label in not_enough_info:
		return 'NOT ENOUGH INFO', 'NOT VERIFIABLE'
	else:
		return 'other', 'NOT VERIFIABLE'

data['label'] = data.apply (lambda row: group_labels(row['original_label'])[0], axis=1)

data['verifiable']= data.apply (lambda row: group_labels(row['original_label'])[1], axis=1)

print(data.groupby('label').count()['id'])





