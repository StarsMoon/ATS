import os
from functools import partial

from paddle.io import Dataset, BatchSampler, DataLoader
from paddlenlp.data import Pad, Tuple
from paddlenlp.datasets import MapDataset, load_dataset

from .utils import load_pickle, save_pickle


class MyDataset(Dataset):
    def __init__(self, ds_name='lcsts', split='train'):
        path_map = {
            'lcsts': os.path.join("caches", f"lcsts_new_{split}_tkned" + ".pkl")
        }
        path = path_map[ds_name]

        def load_data_from_source(path):
            data = load_pickle(path)
            return data

        self.data = load_data_from_source(path)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def trans_func(example, tokenizer, args):
    inputs = example["source"]
    targets = example["target"]

    source = tokenizer(inputs, max_seq_len=args.max_source_length, pad_to_max_seq_len=False,return_attention_mask=True, return_token_type_ids=False,truncation_strategy="longest_first")

    target = tokenizer(targets, max_seq_len=args.max_target_length, pad_to_max_seq_len=False, return_attention_mask=True,return_token_type_ids=False,truncation_strategy="longest_first")

    return (
        source["input_ids"],
        source["attention_mask"],
        source["input_ids"],
        target["input_ids"],
        target["attention_mask"],
        target["input_ids"],
    )


def get_train_dataloader(tokenizer, args):
    filename = os.path.join("caches", "lcsts_new_train" + ".pkl")

    if os.path.exists(filename):
        ds = load_pickle(filename)
    else:
        ds = load_dataset("lcsts_new", splits="train")
        ds.map(
            partial(trans_func, tokenizer=tokenizer, args=args),
            batched=False,
            lazy=False,
        )

    batch_sampler = BatchSampler(ds, batch_size=args.train_batch_size, shuffle=True)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id,dtype="int64"),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_id,dtype="int64"),  # attention_mask
        Pad(axis=0, pad_val=-100,dtype="int64"),  # lm_labels
        Pad(axis=0, pad_val=tokenizer.pad_token_id,dtype="int64"),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_id,dtype="int64"),  # attention_mask
        Pad(axis=0, pad_val=-100,dtype="int64"),  # lm_labels
    ): fn(samples)

    data_loader = DataLoader(
        dataset=ds,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        num_workers=args.num_workers,
        return_list=True,
    )

    return data_loader


def get_dev_dataloader(tokenizer, args):
    filename = os.path.join("caches", "lcsts_new_dev" + ".pkl")

    if os.path.exists(filename):
        ds = load_pickle(filename)
    else:
        ds = load_dataset("lcsts_new", splits="dev")
        ds.map(
            partial(trans_func, tokenizer=tokenizer, args=args),
            batched=False,
            lazy=False,
        )
        
    batch_sampler = BatchSampler(ds, batch_size=args.eval_batch_size, shuffle=False)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id,dtype="int64"),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_id,dtype="int64"),  # attention_mask
        Pad(axis=0, pad_val=-100,dtype="int64"),  # lm_labels
        Pad(axis=0, pad_val=tokenizer.pad_token_id,dtype="int64"),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_id,dtype="int64"),  # attention_mask
        Pad(axis=0, pad_val=-100,dtype="int64"),  # lm_labels
    ): fn(samples)

    data_loader = DataLoader(
        dataset=ds,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        num_workers=args.num_workers,
        return_list=True,
    )

    return data_loader