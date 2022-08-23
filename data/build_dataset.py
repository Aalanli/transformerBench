# %%
from datasets import load_dataset, interleave_datasets
from data.dataset import SimpleBatcher, RemoteSimpleBatcher, SingleActorWrapper


def build_wiki(shuffle=False):
    data_dir = '/media/allan/A/datasets/Huggingface'
    datasets = data_dir + '/datasets'

    wikitext = load_dataset('wikitext', 'wikitext-103-v1', split='train', cache_dir=datasets)
    if shuffle:
        wikitext = wikitext.shuffle(seed=42)
    return wikitext


def build_bert(shuffle=False):
    data_dir = '/media/allan/A/datasets/Huggingface'
    datasets = data_dir + '/datasets'

    wiki = load_dataset("wikipedia", "20220301.en", split="train", cache_dir=datasets)
    wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])
    if shuffle:
        wiki = wiki.shuffle(seed=42)

    bookcorpus = load_dataset('bookcorpus', split='train', cache_dir=datasets)
    if shuffle:
        bookcorpus = bookcorpus.shuffle(seed=42)

    dataset = interleave_datasets([wiki, bookcorpus])
    return dataset


def iter_and_batch(dataset, batch_size=4, seq_len=1024):
    cache_dir = '/media/allan/A/datasets/Huggingface'
    dataset = iter(dataset)
    dataset = map(lambda x: x['text'], dataset)
    batcher = SimpleBatcher(dataset, seq_len, batch_size, cache_dir=cache_dir)
    return batcher


def iter_and_batch_multiprocess(dataset_builder_fn, batch_size=4, seq_len=1024, buffer_size=2):
    cache_dir = '/media/allan/A/datasets/Huggingface'
    def dataset_fn():
        dataset = iter(dataset_builder_fn())
        dataset = map(lambda x: x['text'], dataset)
        return dataset
    
    batcher = RemoteSimpleBatcher.remote(dataset_fn, seq_len, batch_size, cache_dir=cache_dir)
    batcher = SingleActorWrapper(batcher, buffer_size)
    return batcher

# %%
if __name__ == '__main__':
    dataset = build_bert
    batcher = iter_and_batch_multiprocess(dataset)
    print(next(batcher))
