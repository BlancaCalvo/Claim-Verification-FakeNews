### Requirements

```
pip install -r requirement.txt
```

### Evidence retrieval (from the scripts of Athene UKP TU Darmstadt)

Downloaded from GEAR: https://github.com/thunlp/GEAR 

```
data/retrieved/
    train.ensembles.s10.jsonl
    dev.ensembles.s10.jsonl
    test.ensembles.s10.jsonl
```

### Data preparation (from GEAR with modifications)

```
# Download the fever database
wget -O data/fever/fever.db https://s3-eu-west-1.amazonaws.com/fever.public/wiki_index/fever.db

# Extract the evidence from database
python experiment-5/preparation/retrieval_to_bert_input.py

# Build the datasets for gear
python experiment-5/preparation/build_gear_input_set.py

```

### SRL extraction

```
CUDA_VISIBLE_DEVICES=0 python experiment-5/SRL_extraction --input_file data/gear/gear-train-set-0_001.tsv --output_file data/graph_features/train_srl_all.json --cuda 0 

CUDA_VISIBLE_DEVICES=0 python experiment-5/SRL_extraction --input_file data/gear/gear-dev-set-0_001.tsv --output_file data/graph_features/dev_srl_all.json --cuda 0
```

### BERT base model

```
CUDA_VISIBLE_DEVICES=4 python experiment-5/plain_bert/fever_bert.py &> bert_train.log &
```

### SRL model: Sembert with modifications

```
python experiment-5/fever_bert_srl.py --train_srl_file data/srl_features/train_srl_all.json --dev_srl_file data/srl_features/dev_srl_all.json --mapping dream --concat --cuda_devices 0,1,2 --seq_length 250 

python experiment-5/fever_bert_srl.py --train_srl_file data/srl_features/train_trial.json --dev_srl_file data/srl_features/trial.json --concat --cuda_devices 0,1,2 --batch_size 16 --seq_length 100 --max_num_aspect 10
```

## Test

```
PYTHONPATH=experiment-5 python experiment-5/evaluation/test.py --concat --mapping dream --seq_length 200 --max_num_aspect 12
```