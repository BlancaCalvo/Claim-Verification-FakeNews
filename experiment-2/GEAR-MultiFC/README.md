## GEAR-MultiFC
Bsed on the code of the paper "[GEAR: Graph-based Evidence Aggregating and Reasoning for Fact Verification](GEAR.pdf)".

## Requirements:
Please make sure your environment includes:
```
python (tested on 3.6.7)
pytorch (tested on 1.0.0)
```
Then, run the command:
```
pip install -r requirements.txt
```
## Transformation of the MultiFC dataset
The dataset can be downloaded in: https://competitions.codalab.org/competitions/21163 

Clean the dataset:
```
python data/MultiFC/transform/change_format_snippets.py data/MultiFC/snippets/ --output_dir data/MultiFC/new_snippets/
python data/MultiFC/transform/change_format.py data/MultiFC/dev.tsv --output data/MultiFC/changed_dev.tsv
python data/MultiFC/transform/change_format.py data/MultiFC/train.tsv --output data/MultiFC/changed_train.tsv
```

Map the categories, add the evidence and put it in GEAR format. 

Snippets given by MultiFC:
```
python data/MultiFC/transform/multifc_to_gear.py data/MultiFC/changed_dev.tsv --output data/MultiFC/dev_data.tsv
python data/MultiFC/transform/multifc_to_gear.py data/MultiFC/changed_train.tsv --output data/MultiFC/train_data.tsv
```

Reasons to label + snippets for NOTENOUGHINFO:
```
python data/MultiFC/transform/evidence_multi_to_gear.py data/MultiFC/changed_dev.tsv --output data/MultiFC/evidence_dev_data.tsv
python data/MultiFC/transform/evidence_multi_to_gear.py data/MultiFC/changed_train.tsv --output data/MultiFC/evidence_train_data.tsv
```

## Feature Extraction
First download our pretrained BERT-Pair model ([Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/1499a062447f4a3d8de7/?p=/BERT-Pair&mode=list) or [Google Cloud](https://drive.google.com/drive/folders/1y-5VdcrqEEMtU8zIGcREacN1JCHqSp5K)) and put the files into the ``pretrained_models/BERT-Pair/`` folder.

Then the folder will look like this:
```
pretrained_models/BERT-Pair/
    	pytorch_model.bin
    	vocab.txt
    	bert_config.json
```

Then run the feature extraction scripts.
```
chmod +x experiment-2/GEAR-MultiFC/feature_extractor/*.sh
experiment-2/GEAR-MultiFC/feature_extractor/dev_extractor.sh
experiment-2/GEAR-MultiFC/feature_extractor/train_extractor.sh
```

## GEAR Training and Testing
```
CUDA_VISIBLE_DEVICES=0 python experiment-2/GEAR-MultiFC/gear/train.py 
python experiment-2/GEAR-MultiFC/gear/test.py 
python experiment-2/GEAR-MultiFC/gear/evaluation.py 
local:
python experiment-2/GEAR-MultiFC/my_scripts/test.py 
python experiment-2/GEAR-MultiFC/my_scripts/evaluation.py 
```

## Feature extraction, training and testing with new evidences
```
bash experiment-2/GEAR-MultiFC/evidence_stuff/run_all.sh
```
