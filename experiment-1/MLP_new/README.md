
##Sample data: 

python experiment-1/Claim-Verification-FakeNews/MLP_new/scripts/retrieval/document/batch_ir_ns.py --model experiment-1/MLP_new/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --count 1 --split train

##Train with new retrieved data

PYTHONPATH=experiment-1/MLP_new/ python experiment-1/MLP_new/scripts/rte/mlp/train_mlp.py data/fever/fever.db experiment-1/MLP_new/sampled_data/train.ns.pages.p1.jsonl experiment-1/MLP_new/sampled_data/dev.ns.pages.p1.jsonl --model ns_nn_sent --sentence true --features TFIDF

##Train with word embeddings

PYTHONPATH=experiment-1/MLP_new/ python experiment-1/MLP_new/scripts/rte/mlp/train_mlp.py data/fever/fever.db experiment-1/MLP_new/sampled_data/train.ns.pages.p1.jsonl experiment-1/MLP_new/sampled_data/dev.ns.pages.p1.jsonl --model BERT_concat_model --sentence true --features BERT

##Trial script:

PYTHONPATH=experiment-1/MLP_new/ python experiment-1/MLP_new/my_scripts/simple_MLP.py data/fever/fever.db experiment-1/MLP_new/sampled_data/train.ns.pages.p1.jsonl experiment-1/MLP_new/sampled_data/dev.ns.pages.p1.jsonl --model ns_nn_sent 

