chmod +x experiment-2/GEAR-MultiFC/evidence_stuff/feature_extractor/*.sh
experiment-2/GEAR-MultiFC/evidence_stuff/feature_extractor/dev_extractor.sh
experiment-2/GEAR-MultiFC/evidence_stuff/feature_extractor/train_extractor.sh

CUDA_VISIBLE_DEVICES=1 python experiment-2/GEAR-MultiFC/evidence_stuff/gear/train.py 
python experiment-2/GEAR-MultiFC/evidence_stuff/gear/test.py 
python experiment-2/GEAR-MultiFC/evidence_stuff/gear/evaluation.py 
