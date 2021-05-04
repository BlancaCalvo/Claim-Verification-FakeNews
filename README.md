# Claim-Verification-FakeNews

OUTDATED - Experiment 1: it's mainly a copy of the baseline system https://github.com/sheffieldnlp/fever-naacl-2018 , the MLP part. I made some changes so it would take the evidences of UKP-Athenes and changed the features to BERT embeddings (but I didn't get to finish it). 

OUTDATED - Experiment 2: it's the addaptation of the GEAR system (https://github.com/thunlp/GEAR) to the MultiFC dataset. The dataset was first put in the right shape in data/MultiFC/tranform/, which included mapping the labels to FEVER and adding the evidences (I did that both with the snippets and with the fact-checkers explanations - GEAR-MultiFC/evidence_stuff/). There are also new evaluation scripts in GEAR-MultiFC/my_scripts. 

OUTDATED - Experiment 4: I am experimenting with Semantic Roles information to deal with claim complexity and multi-hop (inspired on the DREAM system). Additionally, to inspect the differences between synthetic and naturally-occuring claims, I annotated subsets of FEVER, MultiFC and Hover. I annotated: claim complexity, time reasoning, time complexity and maths reasoning. Results can be seen in visuals/.

ACTUAL THESIS -> Experiment 5: following the work in experiment 4, I trained a model for the FEVER dataset using a BERT model that includes Semantic Role Labels and another that includes OpenIE tuples. The original model is SemBERT (https://github.com/cooelf/SemBERT). I am currently working on the explainability of my models using scripts from https://github.com/copenlu/xai-benchmark. 

See README of each of the experiments in the folders.
