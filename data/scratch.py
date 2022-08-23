# %% 
from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer

from data.map_dataset import MapGenerator, Flatten
from data.dataset import BatchTokenize

from tqdm import tqdm
import ray

data_dir = '/media/allan/A/datasets/Huggingface'
datasets = data_dir + '/datasets'
tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=data_dir)
print('gpt2 tokenizer vocab size:', tokenizer.vocab_size)

# %%

wiki = load_dataset("wikipedia", "20220301.en", split="train", cache_dir=datasets)
wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])
wiki = wiki.shuffle(seed=42)

bookcorpus = load_dataset('bookcorpus', split='train', cache_dir=datasets)
bookcorpus = bookcorpus.shuffle(seed=42)

dataset = interleave_datasets([wiki, bookcorpus])
gen_dataset = iter(dataset)

# %%
token_fn = [BatchTokenize.remote(1024, 4, None, data_dir) for _ in range(8)]
map_fn = map(lambda x: x['text'], gen_dataset)

map_fn = MapGenerator(map_fn, token_fn, 500, 4)
map_fn = map(ray.get, map_fn)
map_fn = Flatten(map_fn)
print(next(map_fn))

gen_dataset = iter(dataset)
tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=data_dir)
for i in tqdm(gen_dataset):
    h = tokenizer(i['text'], truncation=False)

