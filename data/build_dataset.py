# %%
from datasets import load_dataset, interleave_datasets
from data.dataset import SimpleBatcher, RemoteSimpleBatcher, SingleActorWrapper


def build_raw_wiki(shuffle=False):
    data_dir = '/media/allan/A/datasets/Huggingface'
    datasets = data_dir + '/datasets'

    wikitext = load_dataset('wikitext', 'wikitext-103-v1', split='train', cache_dir=datasets)
    if shuffle:
        wikitext = wikitext.shuffle(seed=42)
    return wikitext


def build_raw_bert(shuffle=False):
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


def iter_and_batch_multiprocess(dataset_builder_fn, batch_size=4, seq_len=1024, buffer_size=2):
    cache_dir = '/media/allan/A/datasets/Huggingface'
    def mk_generator_fn(dataset):
        dataset = iter(dataset)
        dataset = map(lambda x: x['text'], dataset)
        return dataset
    
    batcher = RemoteSimpleBatcher.remote(dataset_builder_fn, mk_generator_fn, seq_len, batch_size, cache_dir=cache_dir)
    batcher = SingleActorWrapper(batcher, buffer_size)
    return batcher


def build_bert(batch_size=4, seq_len=1024, buffer_size=2):
    return iter_and_batch_multiprocess(build_raw_bert, batch_size, seq_len, buffer_size)


def build_wiki(batch_size=4, seq_len=1024, buffer_size=2):
    return iter_and_batch_multiprocess(build_raw_wiki, batch_size, seq_len, buffer_size)


# %%
if __name__ == '__main__':
    dataset = build_wiki()
    print(next(dataset))
