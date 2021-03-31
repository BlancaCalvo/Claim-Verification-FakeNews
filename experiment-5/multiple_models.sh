python experiment-5/fever_bert_srl.py --train_srl_file data/srl_features/N_train_srl_all.json --dev_srl_file data/srl_features/N_dev_srl_all.json --cuda_devices 0,1,2 --concat --seq_length 250 --max_num_aspect 12 --batch_size 20
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=experiment-5 python experiment-5/evaluation/test.py --concat --seq_length 250 --max_num_aspect 12 --batch_size 20

python experiment-5/fever_bert_srl.py --train_srl_file data/srl_features/N_train_srl_all.json --dev_srl_file data/srl_features/N_dev_srl_all.json --cuda_devices 0,1,2 --concat --seq_length 250 --max_num_aspect 12 --batch_size 20 --mapping dream

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=experiment-5 python experiment-5/evaluation/test.py --concat --seq_length 250 --max_num_aspect 12 --batch_size 20 --mapping dream

python experiment-5/fever_bert_srl.py --train_srl_file data/srl_features/N_train_srl_all.json --dev_srl_file data/srl_features/N_dev_srl_all.json --cuda_devices 0,1,2 --concat --seq_length 250 --max_num_aspect 12 --batch_size 20 --mapping tags1

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=experiment-5 python experiment-5/evaluation/test.py --concat --seq_length 250 --max_num_aspect 12 --batch_size 20 --mapping tags1

python experiment-5/fever_bert_srl.py --train_srl_file data/srl_features/N_train_srl_all.json --dev_srl_file data/srl_features/N_dev_srl_all.json --cuda_devices 0,1,2 --concat --seq_length 250 --max_num_aspect 12 --batch_size 20 --aggregate

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=experiment-5 python experiment-5/evaluation/test.py --concat --seq_length 250 --max_num_aspect 12 --batch_size 20 --aggregate
