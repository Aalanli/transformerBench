# %%
import random
import ray
import torch
from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer


@ray.remote
class BatchTokenize:
    def __init__(self, seq_len, batch_size, tokenizer=None, cache_dir=None) -> None:
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained('gpt2', cache_dir=cache_dir)
        else:
            self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_len = seq_len
    
    def call(self, x):
        tokenized = self.tokenizer(x, truncation=False)
        tokens = []
        for example in tokenized['input_ids']:
            tokens.append(self.tokenizer.bos_token_id)
            tokens.extend(example)
            tokens.append(self.tokenizer.eos_token_id)
        assert len(tokens) > self.batch_size * self.seq_len, "not enough tokens"
        shift_augment = min(self.seq_len, len(tokens) - self.batch_size * self.seq_len)
        ind_start = random.randint(0, shift_augment - 1)
        tokens = tokens[ind_start:]
        stride = self.batch_size * self.seq_len
        tensors = []
        for i in range(0, len(tokens), stride):
            batch = torch.tensor(tokens[i:i+stride])
            tensors.append(batch.reshape([self.batch_size, self.seq_len]))
        return tensors
