
from allen_srl import run_predictor_batch
from allennlp.predictors import Predictor

import re
import itertools

class InputExample(object):

    def __init__(self, unique_id, text_a, text_b, label, index, is_claim):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b
        print(unique_id)
        print('[CLS]', text_a, '[SEP]', text_b, '[SEP]')
        self.label = label
        self.index = index
        self.is_claim = is_claim

def read_examples(input_file, predictor):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    unique_id = 0
    with open(input_file, "r", encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            line = line.strip().split('\t')
            index = line[0]
            label = line[1]
            claim = line[2]
            evidences = line[3:]

            prediction = run_predictor_batch([{'sentence':claim}], predictor)
            for proposition in prediction[0]['verbs']:
                all_nodes = []
                sr_parts = re.findall(r'\[.*?\]', proposition['description'])
                for part in sr_parts:
                    #role, argument = part.replace('[','').replace(']','').split(': ', 1)
                    node = part.replace('[', '').replace(']', '').replace(':', '')
                    all_nodes.append(node)
                    #if role != 'V':
                    #    examples.append(InputExample(unique_id=unique_id, text_a=role+' '+proposition['verb'], text_b=argument, label=label, index=index,
                    #                 is_claim=True))
                    #    unique_id += 1
                for pair in itertools.combinations(all_nodes, 2):
                    examples.append(InputExample(unique_id=unique_id, text_a=pair[0], text_b=pair[1],
                                 label=label, index=index,is_claim=True))
                    unique_id += 1


            for evidence in evidences:
                evidence = re.sub(r'\.[a-zA-Z0-9 #\-–:]*$', '', evidence) # instead of this line I should change the build_gear_input_set.py script
                prediction = run_predictor_batch([{'sentence': evidence}], predictor)

                for proposition in prediction[0]['verbs']:
                    all_nodes = []
                    sr_parts = re.findall(r'\[.*?\]', proposition['description'])
                    for part in sr_parts:
                        node = part.replace('[', '').replace(']', '').replace(':', '')
                        all_nodes.append(node)
                        #role, argument = part.replace('[', '').replace(']', '').split(': ', 1)
                        #if role != 'V':
                        #    examples.append(InputExample(unique_id=unique_id, text_a=role + ' ' + proposition['verb'], text_b=argument, label=label, index=index,
                        #                                 is_claim=False))
                        #    unique_id += 1
                    for pair in itertools.combinations(all_nodes, 2):
                        examples.append(InputExample(unique_id=unique_id, text_a=pair[0], text_b=pair[1],
                                 label=label, index=index,is_claim=False))
                        unique_id += 1
            break
    return examples

# load the fever dataset with evidences (the GEAR one I think)
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.11.19.tar.gz")

examples = read_examples('data/gear/gear-dev-set-0_001.tsv', predictor)

# for each claim do SRL parsing

# structure the claim so it looks like
#[CLS] V ARG1 became [SEP] Colin Kaepernick [SEP]
#[CLS] V ARG2 became [SEP] a starting quarterback [SEP]
#[CLS] V ARG-TMP became [SEP] during the 49ers 63rd season in the National Football League [SEP]
#[CLS] ARG1 ARG2 Colin Kaepernick [SEP] a starting quarterback [SEP]
#[CLS] ARG1 ARG-TMP Colin Kaepernick [SEP] during the 49ers 63rd season in the National Football League [SEP]
#[CLS]  ARG2 ARG-TMP a starting quarterback  [SEP] during the 49ers 63rd season in the National Football League [SEP]

# adapt it to BERT format so it can be used for fine-tuning with the new vocabulary ver arg arg-tmp arg-loc