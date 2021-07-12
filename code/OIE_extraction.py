import logging
import argparse
from allennlp.predictors import Predictor
import json
from allennlp import predictors
from allennlp.models.archival import load_archive

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def _load_predictor(archive_file: str, predictor_name: str) -> Predictor:
    """
    Helper to load the desired predictor from the given archive.
    """
    archive = load_archive(archive_file)
    return Predictor.from_archive(archive, predictor_name)

def open_information_extraction_stanovsky_2018() -> predictors.OpenIePredictor:
    predictor = _load_predictor(
        "https://allennlp.s3.amazonaws.com/models/openie-model.2020.02.10.tar.gz",
        "open-information-extraction",
    )
    return predictor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default='data/gear/gear-train-set-0_001.tsv', type=str, required=True)
    parser.add_argument("--output_file", default='data/graph_features/oie_features.json', type=str, required=True)
    parser.add_argument("--cuda", default=-1, type=int, required=False) # set to 0
    args = parser.parse_args()
    print(args.input_file)
    print(args.output_file)
    json_list = []
    unique_id = line_n = 0
    predictor = open_information_extraction_stanovsky_2018()

    num_lines = sum(1 for line in open(args.input_file))
    with open(args.input_file, "r", encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            logger.info("Parsing input with OIE: {}/{}".format(line_n, num_lines))
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