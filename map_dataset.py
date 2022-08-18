# %%
from dataclasses import dataclass
from collections import deque
from typing import Generator, List, Callable


@dataclass
class MapGenerator:
    generator: Generator
    actors: List[Callable]
    elems_per_actor: int
    buffer_size: int

    def __post_init__(self):
        self.num_actors = len(self.actors)
        self.queue = deque()
        for i in range(self.buffer_size):
            x, is_end = self.consume_elem()
            if is_end:
                break
            self.queue.append((self.call_actor(self.actors[i % self.num_actors], x), i % self.buffer_size))
    
    def call_actor(self, actor, x):
        if hasattr(actor, 'call'):
            return actor.call.remote(x)
        elif hasattr(actor, 'remote'):
            return actor.remote(x)
        else:
            actor(x)
    
    def consume_elem(self):
        buf = []
        i = 1
        for x in self.generator:
            buf.append(x)
            if i >= self.elems_per_actor:
                break
            i += 1
        return buf, len(buf) == 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if len(self.queue) == 0:
            raise StopIteration
        future, idx = self.queue.popleft()
        x, is_end = self.consume_elem()
        if not is_end:
            self.queue.append((self.call_actor(self.actors[idx], x), idx))
        return future

@dataclass
class FoldGenerator:
    generator: Generator
    fold_fn: Callable
    actors: List[Callable]
    elems_per_actor: int
    buffer_size: int

    def __post_init__(self):
        self.num_actors = len(self.actors)
        self.queue = deque()
        for i in range(self.buffer_size):
            future, is_end = self.fold_fn(self.actors[i % self.num_actors], self.generator)
            if is_end:
                break
            self.queue.append((future, i % self.num_actors))
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if len(self.queue) == 0:
            raise StopIteration
        future, idx = self.queue.popleft()
        x, is_end = self.fold_fn(self.actors[idx], self.generator)
        if not is_end:
            self.queue.append((x, idx))
        return future


@dataclass
class Flatten:
    generator: Generator
    def __post_init__(self):
        self.cur_elem = next(self.generator)
        self.idx = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.idx < len(self.cur_elem):
            self.idx += 1
            return self.cur_elem[self.idx - 1]
        else:
            self.cur_elem = next(self.generator)
            self.idx = 1
            return self.cur_elem[0]


