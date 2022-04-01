## A Semantics-Aware Approach to Automated Claim Verification

This repository has the code of my master thesis "A Semantics-Aware Approach to Automated Claim Verification" at the University of the Basque Country, the University of Groningen and Vicomtech. 

### Abstract

The influence of fake news in the perception of reality has become a mainstream topic in the last years due to the fast propagation of miss-leading information, which has been enhanced by social media. To contribute on the fight against misinformation, researchers have proposed to develop automated solutions. The task of automated claim verification consists in assessing the truthfulness of a claim by finding evidence about its veracity. Datasets with synthetic claims have been developed to train models that perform this task. However, naturally-occurring claims are usually semantically more complex than synthetic claims. In this work, we test if the use of explicit semantic structures can help with the task of claim verification. We introduce Semantic Role Labels and Open Information Extraction structures to a BERT model, showing some improvement on the performance of the task. Additionally, we perform some explainability tests which show that the semantically-enriched model is better at handling complex cases, such as sentences in passive form or with multiple propositions.

### Requirements

```
pip install -r requirements.txt
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
python code/preprocess/retrieval_to_bert_input.py

# Build the datasets for gear
python code/preprocess/build_gear_input_set.py --test --dev --train

```

### BERT base model

```
CUDA_VISIBLE_DEVICES=0 python code/base_bert/fever_bert.py 

CUDA_VISIBLE_DEVICES=0 python code/base_bert/test.py --out dev-results.tsv
```


### SRL extraction

```
CUDA_VISIBLE_DEVICES=0 python code/SRL_extraction.py 
    --input_file data/gear/N_gear-train-set-0_001.tsv 
    --output_file data/srl_features/train_srl_all.json 
    --cuda 0 

CUDA_VISIBLE_DEVICES=0 python code/SRL_extraction.py \
    --input_file data/gear/N_gear-dev-set-0_001.tsv \
    --output_file data/srl_features/dev_srl_all.json \
    --cuda 0

CUDA_VISIBLE_DEVICES=0 python code/SRL_extraction.py 
    --input_file data/gear/examples.tsv 
    --output_file data/srl_features/examples.json 
    --cuda 0

```

### Train and test Sembert with Semantic Role Labels

```
python code/sembert_train.py  
    --train_file data/srl_features/train_srl_all.json 
    --dev_file data/srl_features/dev_srl_all.json 
    --mapping tags1 
    --cuda_devices 0,1,2 
    --seq_length 250 
    --batch_size 20 
    --max_num_aspect 12 

PYTHONPATH=code python code/evaluation/test.py \
    --mapping tags1 \
    --seq_length 250 \
    --max_num_aspect 12 \ 
    --batch_size 20 \
    --dataset data/srl_features/dev_srl_all.json \
    --out dev-results.txt
```

### OpenIE extraction

```
pip install -r oie_requirements.txt

CUDA_VISIBLE_DEVICES=0 python code/OIE_extraction.py 
    --input_file data/gear/N_gear-train-set-0_001.tsv 
    --output_file data/oie_features/train_oie_all.json 
    --cuda 0 

CUDA_VISIBLE_DEVICES=1 python code/OIE_extraction.py 
    --input_file data/gear/N_gear-dev-set-0_001.tsv 
    --output_file data/oie_features/dev_oie_all.json 
    --cuda 0
```

### Train and test Sembert with OpenIE

```
python code/sembert_train.py 
    --train_file data/oie_features/train_oie_all.json 
    --dev_file data/oie_features/dev_oie_all.json 
    --mapping binary 
    --cuda_devices 0,1,2 
    --seq_length 250 
    --batch_size 20 
    --max_num_aspect 12

PYTHONPATH=code python code/evaluation/test.py 
    --mapping binary 
    --seq_length 250 
    --max_num_aspect 12 
    --batch_size 20
```

### FEVER score for dev set

```
python code/evaluation/results_scorer.py 
    --predicted_labels code/outputs/f_sembert-concat_True-agg_False-20batch_size-250seq_length-12n_aspect-tags1/dev-results.tsv
    --predicted_evidence data/gear/bert-nli-dev-retrieve-set.tsv 
    --actual data/fever/shared_task_dev.jsonl

```

### Produce prediction files for test set

```
PYTHONPATH=code python code/evaluation/test.py 
    --mapping tags1 
    --seq_length 250 
    --max_num_aspect 12 
    --batch_size 20 
    --dataset data/srl_features/N_test_srl_all.json 
    --out test-results.txt

python code/evaluation/results_scorer.py 
    --predicted_labels code/outputs/f_sembert-concat_True-agg_False-20batch_size-250seq_length-12n_aspect-tags1/test-results.tsv 
    --predicted_evidence data/gear/bert-nli-test-retrieve-set.tsv --test

CUDA_VISIBLE_DEVICES=0 python code/base_bert/test.py 
    --dev_features data/gear/N_gear-test-set-0_001.tsv 
    --out test-results.tsv

python code/evaluation/results_scorer.py 
    --predicted_labels code/outputs/F-base-bert-2/test-results.tsv 
    --predicted_evidence data/gear/bert-nli-test-retrieve-set.tsv 
    --shared_task data/fever/shared_task_test.jsonl 
    --test

```

### Saliency Scores BERT and SemBERT models

Modified from: https://github.com/copenlu/xai-benchmark

```
PYTHONPATH=code/ python code/saliency/interpret_grads_occ.py 
    --models_dir code/outputs/F-base-bert-2/ 
    --dataset_dir data/X_tests/examples.tsv 
    --saliency guided sal inputx 
    --model trans 
    --dataset fever 
    --no_time

PYTHONPATH=code/ python code/saliency/sembert_interpret_grads.py 
    --dev_file data/srl_features/examples.json 
    --saliency guided sal inputx
```

### Testing with adversarial attacks

```
python code/SRL_extraction.py \
    --input_file data/X_tests/adversarial_examples.tsv \
    --output_file data/srl_features/adversarial_examples.json

PYTHONPATH=code python code/evaluation/test.py \
    --mapping tags1 \
    --seq_length 250 \
    --max_num_aspect 12 --batch_size 20 \
    --dataset data/srl_features/adversarial_examples.json \
    --out adver-results.tsv --no_cuda

PYTHONPATH=code python code/base_bert/test.py \
    --dev_features data/X_tests/adversarial_examples.tsv \
    --out adver-results.tsv --no_cuda
```

### Second Phase: Semantically complex claims and evidences

```
python code/saliency/filter_complex_claims.py

python code/SRL_extraction.py \
    --input_file data/X_tests/semantically_complex_evidence.tsv \
    --output_file data/srl_features/semantically_complex_evidence.json

PYTHONPATH=code python code/evaluation/test.py \
    --mapping tags1 \
    --seq_length 250 \
    --max_num_aspect 12 \
    --batch_size 20 \
    --dataset data/srl_features/semantically_complex_evidence.json \
    --out semantically_complex_evidence.tsv \
    --no_cuda

PYTHONPATH=code python code/base_bert/test.py \
    --dev_features data/X_tests/semantically_complex_evidence.tsv \
    --out semantically_complex_evidence.tsv --no_cuda
    
python code/evaluation/get_acc.py
```
