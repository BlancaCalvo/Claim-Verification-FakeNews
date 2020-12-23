from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive
from contextlib import ExitStack
import argparse
import json

def run_predictor_batch(batch_data, predictor, output_file, print_to_console):
    if len(batch_data) == 1:
        result = predictor.predict_json(batch_data[0])
        results = [result]
    else:
        results = predictor.predict_batch_json(batch_data)

    for model_input, output in zip(batch_data, results):
        string_output = predictor.dump_line(output)
        if print_to_console:
            print("input: ", model_input)
            print("batch data prediction: ", string_output)
        if output_file:
            output_file.write(string_output)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-file', type=argparse.FileType('w'), help='path to output file')
    parser.add_argument('--batch-size', type=int, default=1, help='The batch size to use for processing')
    args = parser.parse_args()
    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.11.19.tar.gz")
    output_file = None
    print_to_console = False

    with ExitStack() as stack:
        if args.output_file:
            output_file = stack.enter_context(args.output_file)
        if not args.output_file:
            print_to_console = True
        batch_data = [{'sentence': 'Colin Kaepernick became a starting quarterback during the 49ers 63rd season in the National Football League.'},
                      {'sentence': 'He remained the team \'s starting quarterback for the rest of the season and went on to lead the 49ers to their first Super Bowl appearance since 1994 , losing to the Baltimore Ravens . quarterback quarterback Super Bowl Super Bowl XLVII 1994 Super Bowl XXIX Baltimore Ravens Baltimore Ravens'},
                      {'sentence': 'Kaepernick began his professional career as a backup to Alex Smith , but became the 49ers \' starter in the middle of the 2012 season after Smith suffered a concussion . Alex Smith Alex Smith 2012 season 2012 San Francisco 49ers season concussion concussion'},
                      {'sentence': 'During the 2013 season , his first full season as a starter , Kaepernick helped the 49ers reach the NFC Championship , losing to the Seattle Seahawks . 2013 season 2013 San Francisco 49ers season NFC Championship 2013–14 NFL playoffs#NFC Championship Game: Seattle Seahawks 23.2C San Francisco 49ers 17 Seattle Seahawks Seattle Seahawks'},
                      {'sentence': 'In the following seasons , Kaepernick lost and won back his starting job , with the 49ers missing the playoffs for three years consecutively .'},
                      {'sentence': 'Colin Rand Kaepernick -LRB- -LSB- ` kæpərnɪk -RSB- ; born November 3 , 1987 -RRB- is an American football quarterback who is currently a free agent . American football American football quarterback quarterback'}]
        run_predictor_batch(batch_data, predictor,
            output_file,
            print_to_console)

if __name__ == '__main__':
    main()
