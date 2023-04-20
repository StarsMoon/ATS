import math
import torch
from torch.utils.data import Dataset
import datasets


class MyDataset(Dataset):
    def __init__(self, tokenizer, batch_size, data_path, features, split='train', keep_in_memory=False):
        raw_datasets = datasets.DatasetDict()
        raw_datasets = raw_datasets.load_from_disk(data_path, keep_in_memory=keep_in_memory)
        self.data = raw_datasets[split]
        self.tkn = tokenizer
        self.batch_size = batch_size
        self.features = features
        
    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError('index {} is out of range'.format(idx))
        return trans_func(self.data[idx * self.batch_size: (idx+1) * self.batch_size], self.tkn, self.features)

    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)
    
    def get_batches(self, nb_batches):
        batches = []
        for i in range(nb_batches):
            batches.append(self[i])
        return batches
    
    def shuffle(self, seed=None):
        self.data.shuffle(seed)


def trans_func(example, tokenizer, features):
    inputs = example[features[0]]
    targets = example[features[1]]

    source = tokenizer.batch_encode_plus(inputs, padding=True, return_tensors='pt')
    target = tokenizer.batch_encode_plus(targets, padding=True, return_tensors='pt')
    return (
        source["input_ids"].cuda(),
        source["attention_mask"].cuda(),
        torch.where(source["input_ids"] > 0, source["input_ids"], -100).cuda(),
        target["input_ids"].cuda(),
        target["attention_mask"].cuda(),
        torch.where(target["input_ids"] > 0, target["input_ids"], -100).cuda(),
    )