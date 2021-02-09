from datasets import Dataset, load_dataset
from transformers import BertForSequenceClassification, BertTokenizer
from extractor import convert_examples_to_features, InputExample
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler


def read_examples(input_file):
    """Read input to dictionary."""
    #examples_dict = {}
    examples = []
    unique_id = 0
    #indexes = []
    #labels = []
    #claims = []
    #ids = []
    #evidences_list=[]
    with open(input_file, "r", encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            line = line.strip().split('\t')

            index = line[0]
            label = line[1]
            claim = line[2]
            evidences = line[3:]

            for evidence in evidences:
                #indexes.append(index)
                #labels.append(label)
                #claims.append(line)
                #evidences_list.append(evidence)
                #ids.append(unique_id)

                unique_id += 1

        #examples_dict = {'index':indexes, 'unique_id':ids, 'label':labels, 'text_a':claims, 'text_b':evidences_list}
                examples.append(InputExample(unique_id=unique_id, text_a=evidence, text_b=claim, label=label, index=index, is_claim=False))

                #examples_dict[unique_id] = {'index': index, 'label': label, 'text_a': claim, 'text_b': evidence}
    return examples


def features_to_torch_dataset(features):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.input_type_ids for f in features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
    eval_sampler = SequentialSampler(eval_data)  # returns indices
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=150)

    return eval_dataloader

train_dataset = read_examples('data/gear/train_trial.tsv')
dev_dataset = read_examples('data/gear/trial.tsv')

#dataset = Dataset.from_dict(dataset)

#print(dataset.features)
#print(dataset.shape)

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=3)

train_features = convert_examples_to_features(train_dataset, 70, tokenizer)
dev_features = convert_examples_to_features(dev_dataset, 70, tokenizer)

train_dataloader = features_to_torch_dataset(train_features)
dev_dataloader = features_to_torch_dataset(dev_features)

#model.eval()


