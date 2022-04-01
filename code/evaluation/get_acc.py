import csv
import numpy as np
from sklearn.metrics import accuracy_score

#file = open('data/X_tests/semantically_complex_claim.csv')
#read_tsv = csv.reader(file)
#true_labels = [line[1] for line in read_tsv]

file = open('code/outputs/F-base-bert-2/semantically_complex_evidence.tsv')
read_tsv = csv.reader(file, delimiter='\t')
true_labels = []
predicted_labels = []
for line in read_tsv:
    predicted_labels.append(np.argmax(line[0:3]))
    true_labels.append(int(line[3]))

file = open('code/outputs/f_sembert-concat_True-agg_False-20batch_size-250seq_length-12n_aspect-tags1/semantically_complex_evidence.tsv')
read_tsv = csv.reader(file, delimiter='\t')
predicted_labels_2 = [np.argmax(line[0:3]) for line in read_tsv]

print(accuracy_score(true_labels, predicted_labels))
print(accuracy_score(true_labels, predicted_labels_2))