# %% 
from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer

data_dir = '/media/allan/A/datasets/Huggingface'

wikitext = load_dataset('wikitext', 'wikitext-103-v1', cache_dir=data_dir + '/datasets')

wikitext['train'][:3]['text']

#data_dir = '/media/allan/A/datasets/Huggingface/datasets'
#
#wikitext = load_dataset('wikitext', 'wikitext-103-v1', split='train', cache_dir=data_dir)
#wiki = load_dataset("wikipedia", "20220301.en", split="train", cache_dir=data_dir)


# %%
wikitext = wikitext.shuffle(seed=42)

wiki = load_dataset("wikipedia", "20220301.en", split="train")
wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])
wiki = wiki.shuffle(seed=42)

bookcorpus = load_dataset('bookcorpus', split='train')
bookcorpus = bookcorpus.shuffle(seed=42)

dataset = interleave_datasets([wiki, bookcorpus])

tokenizer = AutoTokenizer.from_pretrained("gpt2")

def map_fn(examples):
    tokenized = tokenizer(examples['text'], truncation=False)
    tokens = []
    for example in tokenized['input_ids']:
        print(example)
        tokens.append(tokenizer.bos_token_id)
        tokens.extend(example)
        tokens.append(tokenizer.eos_token_id)
    return {'input_ids': tokens}

# %%
dataset = dataset.map(map_fn, batched=True)


# %%
a = tokenizer(
    [wiki[i]['text'] for i in range(0, 10)], 
    truncation=False,
    )
print(a)

# %%
tokenizer.bos_token_id
# %%
dataset = dataset.map(lambda x: tokenizer(x['text']), batched=True)

# %%
print(dataset['train'][3])
# %%
book_dataset = load_dataset('bookcorpus', split='train', streaming=True)
book_dataset = book_dataset.shuffle(seed=1, buffer_size=10000)

