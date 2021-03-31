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
python experiment-5/preprocess/retrieval_to_bert_input.py

# Build the datasets for gear
python experiment-5/preprocess/build_gear_input_set.py

```

### SRL extraction

```
CUDA_VISIBLE_DEVICES=0 python experiment-5/SRL_extraction.py --input_file data/gear/N_gear-train-set-0_001.tsv --output_file data/srl_features/train_srl_all.json --cuda 0 

CUDA_VISIBLE_DEVICES=1 python experiment-5/SRL_extraction.py --input_file data/gear/N_gear-dev-set-0_001.tsv --output_file data/srl_features/N_dev_srl_all.json --cuda 0
```

### BERT base model

```
CUDA_VISIBLE_DEVICES=0 python experiment-5/base_bert/fever_bert.py &> bert_train.log &

python experiment-5/base_bert/test.py
```

### SRL model: Sembert with modifications

```
python experiment-5/fever_bert_srl.py --train_srl_file data/srl_features/train_srl_all.json --dev_srl_file data/srl_features/dev_srl_all.json --mapping dream --concat --cuda_devices 0,1,2 --seq_length 250 

```

## Test Sembert

```
PYTHONPATH=experiment-5 python experiment-5/evaluation/test.py --concat --aggregate --mapping tags1 --seq_length 250 --max_num_aspect 12 --batch_size 20
```

