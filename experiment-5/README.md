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
python experiment-5/preprocess/build_gear_input_set.py --test --dev --train

```

### BERT base model

```
CUDA_VISIBLE_DEVICES=0 python experiment-5/base_bert/fever_bert.py 

CUDA_VISIBLE_DEVICES=0 python experiment-5/base_bert/test.py --out dev-results.tsv
```


### SRL extraction

```
CUDA_VISIBLE_DEVICES=0 python experiment-5/SRL_extraction.py 
    --input_file data/gear/N_gear-train-set-0_001.tsv 
    --output_file data/srl_features/train_srl_all.json 
    --cuda 0 

CUDA_VISIBLE_DEVICES=0 python experiment-5/SRL_extraction.py --input_file data/gear/gold_gear-dev-set-0_001.tsv --output_file data/srl_features/gold_dev_srl_all.json --cuda 0

CUDA_VISIBLE_DEVICES=0 python experiment-5/SRL_extraction.py --input_file data/gear/examples.tsv --output_file data/srl_features/examples.json --cuda 0

```

### Train and test Sembert with Semantic Role Labels

```
python experiment-5/sembert_train.py  --train_file data/srl_features/N_train_srl_all.json --dev_file data/srl_features/N_dev_srl_all.json --mapping tags1 --cuda_devices 0,1,2 --seq_length 250 --batch_size 20 --max_num_aspect 12 

python experiment-5/sembert_train.py  --train_file data/srl_features/gold_train_srl_all.json --dev_file data/srl_features/gold_dev_srl_all.json --mapping tags1 --cuda_devices 1 --seq_length 50 --batch_size 20 --max_num_aspect 4

PYTHONPATH=experiment-5 python experiment-5/evaluation/test.py --mapping tags1 --seq_length 250 --max_num_aspect 12 --batch_size 20 --dataset data/srl_features/dev_srl_all.json --out dev-results.txt
```

### OpenIE extraction

```
pip install -r oie_requirements.txt

CUDA_VISIBLE_DEVICES=0 python experiment-5/OIE_extraction.py 
    --input_file data/gear/N_gear-train-set-0_001.tsv 
    --output_file data/oie_features/train_oie_all.json 
    --cuda 0 

CUDA_VISIBLE_DEVICES=1 python experiment-5/OIE_extraction.py 
    --input_file data/gear/N_gear-dev-set-0_001.tsv 
    --output_file data/oie_features/N_dev_oie_all.json 
    --cuda 0
```

### Train and test Sembert with OpenIE

```
python experiment-5/sembert_train.py --train_file data/oie_features/train_oie_all.json --dev_file data/oie_features/dev_oie_all.json --mapping binary --cuda_devices 0,1,2 --seq_length 250 --batch_size 20 --max_num_aspect 12

PYTHONPATH=experiment-5 python experiment-5/evaluation/test.py 
    --mapping binary 
    --seq_length 250 
    --max_num_aspect 12 
    --batch_size 20
```

### FEVER score

```
python experiment-5/evaluation/results_scorer.py 
    --predicted_labels experiment-5/outputs/f_sembert-concat_True-agg_False-20batch_size-250seq_length-12n_aspect-tags1/dev-results.tsv
    --predicted_evidence data/gear/bert-nli-dev-retrieve-set.tsv 
    --actual data/fever/shared_task_dev.jsonl

python experiment-5/evaluation/results_scorer.py --predicted_labels experiment-5/outputs/f_sembert-concat_True-agg_False-20batch_size-250seq_length-12n_aspect-tags1/test-results.tsv --predicted_evidence data/gear/bert-nli-test-retrieve-set.tsv --shared_task data/fever/shared_task_dev.jsonl --test


```

### Saliency Scores BERT model 

Modified from: https://github.com/copenlu/xai-benchmark

```
PYTHONPATH=experiment-5/ python experiment-5/saliency/interpret_grads_occ.py --models_dir experiment-5/outputs/F-base-bert-2/ --dataset_dir data/gear/examples.tsv --saliency guided sal
 inputx --model trans --dataset fever --no_time
```

### Saliency Scores SemBERT model

```
PYTHONPATH=experiment-5/ python experiment-5/saliency/sembert_interpret_grads.py --dev_file data/srl_features/examples.json --saliency guided sal inputx
```

### Testing with adversarial attacks

```
CUDA_VISIBLE_DEVICES=0 python experiment-5/SRL_extraction.py --input_file data/gear/adversarial_examples.tsv --output_file data/srl_features/adversarial_examples.json --cuda 0

PYTHONPATH=experiment-5 python experiment-5/evaluation/test.py --mapping tags1 --seq_length 250 --max_num_aspect 12 --batch_size 20 --dataset data/srl_features/adversarial_examples.json --out adver-results.tsv

CUDA_VISIBLE_DEVICES=0 python experiment-5/base_bert/test.py --dev_features data/gear/adversarial_examples.tsv --out adver-results.tsv
```

### Testing with UKP-Snopes 

```
python experiment-5/preprocess/snopes_retrieval.py

python experiment-5/preprocess/build_gear_input_set.py --snopes

CUDA_VISIBLE_DEVICES=0 python experiment-5/SRL_extraction.py --input_file data/gear/N_gear-snopes-dev.tsv --output_file data/srl_features/snopes_dev.json --cuda 0

PYTHONPATH=experiment-5 python experiment-5/evaluation/test.py --mapping tags1 --seq_length 250 --max_num_aspect 12 --batch_size 20 --dataset data/srl_features/snopes_dev.json --out snopes-results.tsv

```

