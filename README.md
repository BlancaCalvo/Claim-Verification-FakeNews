# Claim-Verification-FakeNews

Experiment 1: it's mainly a copy of the baseline system https://github.com/sheffieldnlp/fever-naacl-2018 , the MLP part. I made some changes so it would take the evidences of UKP-Athenes and changed the features to BERT embeddings (but I didn't get to finish it). 

Experiment 2: it's the addaptation of the GEAR system (https://github.com/thunlp/GEAR) to the MultiFC dataset. The dataset was first put in the right shape in data/MultiFC/tranform/, which included mapping the labels to FEVER and adding the evidences (I did that both with the snippets and with the fact-checkers explanations - GEAR-MultiFC/evidence_stuff/). There are also new evaluation scripts in GEAR-MultiFC/my_scripts. 

Experiment 4: I am experimenting with Semantic Roles information to deal with claim complexity and multi-hop (inspired on the DREAM system). Additionally, to inspect the differences between synthetic and naturally-occuring claims, I annotated subsets of FEVER, MultiFC and Hover. I annotated: claim complexity, time reasoning, time complexity and maths reasoning. Results can be seen in visuals/.

See README of each of the experiments in the folders.
