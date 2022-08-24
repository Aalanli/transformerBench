# %%
import random
import ray
import torch
import numpy as np
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
        for i in range(0, len(tokens) // stride):
            batch = torch.tensor(tokens[i*stride:(i+1)*stride])
            tensors.append(batch.reshape([self.batch_size, self.seq_len]))
        return tensors


class SimpleBatcher:
    def __init__(self, generator, seq_len, batch_size, tokenizer=None, cache_dir=None) -> None:
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained('gpt2', cache_dir=cache_dir)
        else:
            self.tokenizer = tokenizer
        self.generator = generator
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.elems = batch_size * seq_len
        self.vocab_size = self.tokenizer.vocab_size
    
    def __iter__(self):
        return self
    
    def __next__(self):
        tokens = []
        while len(tokens) < self.elems:
            x = self.tokenizer(next(self.generator), truncation=False)['input_ids']
            tokens.extend(x)
        tokens = tokens[:self.elems]
        tokens = torch.tensor(tokens)
        tokens = tokens.reshape([self.batch_size, self.seq_len])
        return tokens[:, :-1], tokens[:, 1:]
    

@ray.remote
class RemoteSimpleBatcher:
    def __init__(self, raw_dataset_fn, mk_generator_fn, seq_len, batch_size, tokenizer=None, cache_dir=None) -> None:
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained('gpt2', cache_dir=cache_dir)
        else:
            self.tokenizer = tokenizer
        self.raw_dataset = raw_dataset_fn()
        self.iter_fn = mk_generator_fn
        self.generator = mk_generator_fn(self.raw_dataset)

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.elems = batch_size * seq_len
        self.vocab_size = self.tokenizer.vocab_size
    
    def reset(self):
        self.generator = self.iter_fn(self.raw_dataset)
    
    def get(self):
        tokens = []
        while len(tokens) < self.elems:
            x = self.tokenizer(next(self.generator), truncation=False)['input_ids']
            tokens.extend(x)
        tokens = tokens[:self.elems]
        tokens = np.array(tokens)
        tokens = tokens.reshape([self.batch_size, self.seq_len])
        return tokens[:, :-1], tokens[:, 1:]


class SingleActorWrapper:
    """
    A simple iterator wrapper around a single remote actor, for cases
    where dataset fetches are faster than model updates, as theoretically
    only one actor is needed
    """
    def __init__(self, actor, buffer=2) -> None:
        self.actor = actor
        self.buffer = [self.actor.get.remote() for _ in range(buffer)]
    
    def __iter__(self):
        self.actor.reset.remote()
        return self

    def __next__(self):
        future, self.buffer = ray.wait(self.buffer)
        self.buffer.append(self.actor.get.remote())
        return ray.get(future[0])

