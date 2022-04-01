
from collections import Counter
import nltk
import csv

list_counts = []
tsv_file = open("data/gear/N_gear-dev-set-0_001.tsv")
read_tsv = csv.reader(tsv_file, delimiter="\t")

with open("data/X_tests/semantically_complex_evidence.tsv", "w") as f:
    writer = csv.writer(f, delimiter='\t')
    for line in read_tsv:
        #complex=False
        #for p in line[3:5]:
        #tokens = nltk.word_tokenize(line[2].lower()) # claim
        tokens = nltk.word_tokenize(line[3].lower())  # evidence
        text = nltk.Text(tokens)
        tags = nltk.pos_tag(text, tagset='universal')
        counts = Counter(tag for word, tag in tags)
        list_counts.append(counts['VERB'])
        # FOR CLAIMS Counter({1: 10845, 2: 6684, 3: 1703, 0: 375, 4: 279, 5: 44, 6: 12, 7: 6})
        # FOR FIRST EVIDENCES Counter({2: 5511, 3: 5153, 4: 3097, 1: 2563, 5: 1824, 6: 991, 7: 364, 8: 239, 9: 69, 0: 48, 12: 29, 10: 27, 13: 23, 11: 8, 19: 1, 14: 1})
        if counts['VERB'] >= 5:
            #complex = True
    #if complex:
            writer.writerow(line)

print(Counter(list_counts))

