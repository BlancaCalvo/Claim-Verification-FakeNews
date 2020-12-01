CUDA_VISIBLE_DEVICES=1 PYTHONPATH=experiment-2/GEAR-MultiFC/feature_extractor/ python experiment-2/GEAR-MultiFC/feature_extractor/extractor.py \
    --input_file data/MultiFC/evidence_dev_data.tsv \
    --output_file data/MultiFC/evidence_dev_data-features.tsv \
    --bert_model experiment-2/GEAR-MultiFC/pretrained_models/BERT-Pair/ \
    --do_lower_case \
    --max_seq_length 128 \
    --batch_size 512 \
