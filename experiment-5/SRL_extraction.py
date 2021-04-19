
import logging
import argparse
from allennlp.predictors import Predictor
import json

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default='data/gear/gear-train-set-0_001.tsv', type=str, required=True)
    parser.add_argument("--output_file", default='data/graph_features/srl_features.json', type=str, required=True)
    parser.add_argument("--cuda", default=-1, type=int, required=False) # set to 0
    args = parser.parse_args()
    print(args.input_file)
    print(args.output_file)
    json_list = []
    unique_id = line_n = 0
    predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.11.19.tar.gz", cuda_device=args.cuda)

    num_lines = sum(1 for line in open(args.input_file))
    with open(args.input_file, "r", encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            logger.info("Parsing input with SRL: {}/{}".format(line_n, num_lines))
            line_n += 1
            if not line:
                break
            line = line.strip().split('\t')
            index = line[0]
            label = line[1]
            claim = line[2]
            evidences = line[3:]
            claim_prediction = predictor.predict_json({'sentence': claim})

            for evidence in evidences:
                #evidence = re.sub(r'\.[a-zA-Z \-é0-9\(\)]*$', '', evidence)  # this removed NE in a previous version
                try:
                    prediction = predictor.predict_json({'sentence': evidence})
                except RuntimeError:
                    print('Length issue with this evidence: ', evidence)
                    continue

                json_list.append(
                    {'unique_id': unique_id, 'claim': claim, 'claim_srl': claim_prediction, 'evidence': evidence, 'evidence_srl': prediction,
                     'label': label, 'index': index})
                unique_id += 1

    with open(args.output_file, 'w') as fout:
        json.dump(json_list, fout)