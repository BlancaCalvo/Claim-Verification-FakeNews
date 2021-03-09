requirements

pip install -r requirement.txt

base-bert

CUDA_VISIBLE_DEVICES=4 python experiment-5/plain_bert/fever_bert.py &> bert_train.log &

SRL extraction

CUDA_VISIBLE_DEVICES=0 python experiment-5/SRL_extraction --input_file data/gear/gear-train-set-0_001.tsv --output_file data/graph_features/train_srl_all.json --cuda 0 

CUDA_VISIBLE_DEVICES=0 python experiment-5/SRL_extraction --input_file data/gear/gear-dev-set-0_001.tsv --output_file data/graph_features/dev_srl_all.json --cuda 0

fever_bert_srl

CUDA_VISIBLE_DEVICES=0,1,2 python experiment-5/fever_bert_srl.py --train_srl_file data/srl_features/train_srl_all.json --dev_srl_file data/srl_features/dev_srl_all.json --concat --cuda_devi
ces 0,1,2 --batch_size 16 --seq_length 250 &> sembert_concat_train.log &

CUDA_VISIBLE_DEVICES=0,1,2 python experiment-5/fever_bert_srl.py --train_srl_file data/srl_features/train_trial.json --dev_srl_file data/srl_features/trial.json --concat --cuda_devices 0,1,2 --batch_size 16 --seq_length 100 --max_num_aspect 10

test

PYTHONPATH=experiment-5 python experiment-5/evaluation/test.py --vote &> sembert_vote_test.log &

{'verbs': [{'verb': 'born', 'description': "[ARG1: Katherine Matilda ` ` Tilda '' Swinton] -LRB- [V: born] [ARGM-TMP: 5 November 1960] -RRB- is a British actress , performance artist , model , and fashion muse , known for her roles in independent and Hollywood films", 'tags': ['B-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'O', 'B-V', 'B-ARGM-TMP', 'I-ARGM-TMP', 'I-ARGM-TMP', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']}, {'verb': 'is', 'description': "[ARG1: Katherine Matilda ` ` Tilda '' Swinton -LRB- born 5 November 1960 -RRB-] [V: is] [ARG2: a British actress , performance artist , model , and fashion muse , known for her roles in independent and Hollywood films]", 'tags': ['B-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'B-V', 'B-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG2']}, {'verb': 'known', 'description': "Katherine Matilda ` ` Tilda '' Swinton -LRB- born 5 November 1960 -RRB- is [ARG1: a British actress , performance artist , model , and fashion muse] , [V: known] [ARG2: for her roles in independent and Hollywood films]", 'tags': ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'O', 'B-V', 'B-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG2']}], 'words': ['Katherine', 'Matilda', '`', '`', 'Tilda', "''", 'Swinton', '-LRB-', 'born', '5', 'November', '1960', '-RRB-', 'is', 'a', 'British', 'actress', ',', 'performance', 'artist', ',', 'model', ',', 'and', 'fashion', 'muse', ',', 'known', 'for', 'her', 'roles', 'in', 'independent', 'and', 'Hollywood', 'films']}
