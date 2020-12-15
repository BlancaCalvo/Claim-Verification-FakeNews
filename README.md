# Claim-Verification-FakeNews

Experiment 1: it's mainly a copy of https://github.com/sheffieldnlp/fever-naacl-2018 , the MLP part. I made some changes so it would take the evidences of UKP-Athenes and changed the features to BERT embeddings (but I didn't ge tto finish it). 

Experiment 2: it's the addaptation of the GEAR system (https://github.com/thunlp/GEAR) to the MultiFC dataset. The dataset was first put in the right shape in data/MultiFC/tranform/, which included mapping the labels to FEVER and adding the evidences (I did that both with the snippets and with the fact-checkers explanations - GEAR-MultiFC/evidence_stuff/). There are also new evaluation scripts in GEAR-MultiFC/my_scripts. 

Experiment 4: I want to add Semantic Roles information to deal with claim complexity and multi-hop (like the DREAM system). I am first going to annotate subsets of FEVER, MultiFC and Hover. 

See README of each of the experiments in the folders.
