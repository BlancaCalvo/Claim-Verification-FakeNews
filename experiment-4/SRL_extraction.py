import itertools
import logging
import re
from extractor import InputExample

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def read_examples_SRL(input_file, predictor):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    json_list = []
    unique_id = line_n = 0
    num_lines = sum(1 for line in open(input_file))
    with open(input_file, "r", encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            line_n += 1
            logger.info("Parsing input with SRL: {}/{}".format(line_n, num_lines))
            if not line:
                break
            line = line.strip().split('\t')
            index = line[0]
            label = line[1]
            claim = line[2]
            evidences = line[3:]

            prediction = predictor.predict_json({'sentence': claim})
            if len(prediction['verbs']) == 0:
                examples.append(InputExample(unique_id=unique_id, text_a=claim, text_b=None,
                                             label=label, index=index, is_claim=True))
                json_list.append({'unique_id':unique_id, 'text_a':claim, 'text_b':None,
                                             'label':label, 'index':index, 'is_claim':True})
                unique_id += 1
            else:
                for proposition in prediction['verbs']:
                    all_nodes = []
                    sr_parts = re.findall(r'\[[A-Z0-9]+:.*?\]', proposition['description'])
                    for part in sr_parts:
                        role, argument = part.replace('[','').replace(']','').split(': ', 1)
                        all_nodes.append(argument)
                    for pair in itertools.combinations(all_nodes, 2):
                        examples.append(InputExample(unique_id=unique_id, text_a=pair[0], text_b=pair[1],
                                                     label=label, index=index,is_claim=True))
                        json_list.append({'unique_id': unique_id, 'text_a': pair[0], 'text_b': pair[1],
                                          'label': label, 'index': index, 'is_claim': True})
                        unique_id += 1


            for evidence in evidences:
                evidence = re.sub(r'\.[a-zA-Z \-é0-9\(\)]*$', '', evidence) # instead of this line I should change the build_gear_input_set.py script
                try:
                    prediction = predictor.predict_json({'sentence': evidence})
                except RuntimeError:
                    print('Length issue with this evidence: ', evidence)
                    continue
                if len(prediction['verbs']) == 0:
                    continue
                for proposition in prediction['verbs']:
                    all_nodes = []
                    sr_parts = re.findall(r'\[[A-Z0-9]+:.*?\]', proposition['description'])
                    for part in sr_parts:
                        try:
                            role, argument = part.replace('[', '').replace(']', '').split(': ', 1)
                        except ValueError:
                            print('no role')
                            continue
                        all_nodes.append(argument)
                    for pair in itertools.combinations(all_nodes, 2):
                        examples.append(InputExample(unique_id=unique_id, text_a=pair[0], text_b=pair[1],
                                 label=label, index=index,is_claim=False))
                        json_list.append({'unique_id': unique_id, 'text_a': pair[0], 'text_b': pair[1],
                                          'label': label, 'index': index, 'is_claim': False})
                        unique_id += 1
    return examples, json_list

def read_examples_SRL_1claim(input_file, predictor):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    json_list = []
    unique_id = line_n = 0
    num_lines = sum(1 for line in open(input_file))
    with open(input_file, "r", encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            line_n += 1
            logger.info("Parsing input with SRL: {}/{}".format(line_n, num_lines))
            if not line:
                break
            line = line.strip().split('\t')
            index = line[0]
            label = line[1]
            claim = line[2]
            evidences = line[3:]

            examples.append(InputExample(unique_id=unique_id, text_a=claim, text_b=None,
                                             label=label, index=index, is_claim=True))
            json_list.append({'unique_id': unique_id, 'text_a': claim, 'text_b': None,
                              'label': label, 'index': index, 'is_claim': True})
            unique_id += 1

            for evidence in evidences:
                evidence = re.sub(r'\.[a-zA-Z \-é0-9\(\)]*$', '', evidence) # instead of this line I should change the build_gear_input_set.py script
                try:
                    prediction = predictor.predict_json({'sentence': evidence})
                except RuntimeError:
                    print('Length issue with this evidence: ', evidence)
                    continue
                if len(prediction['verbs']) == 0:
                    continue
                for proposition in prediction['verbs']:
                    all_nodes = []
                    sr_parts = re.findall(r'\[[A-Z0-9]+:.*?\]', proposition['description'])
                    for part in sr_parts:
                        try:
                            role, argument = part.replace('[', '').replace(']', '').split(': ', 1)
                        except ValueError:
                            print('no role')
                            continue
                        all_nodes.append(argument)
                    for pair in itertools.combinations(all_nodes, 2):
                        examples.append(InputExample(unique_id=unique_id, text_a=pair[0], text_b=pair[1],
                                 label=label, index=index,is_claim=False))
                        json_list.append({'unique_id': unique_id, 'text_a': pair[0], 'text_b': pair[1],
                                          'label': label, 'index': index, 'is_claim': False})
                        unique_id += 1
    return examples, json_list