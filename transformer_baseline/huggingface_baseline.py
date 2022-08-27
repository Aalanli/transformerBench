# %%
from datasets import load_dataset, interleave_datasets

data_dir = '/media/allan/A/datasets/Huggingface'
datasets = data_dir + '/datasets'

shuffle = False
wikitext = load_dataset('wikitext', 'wikitext-103-v1', cache_dir=datasets, streaming=False)
if shuffle:
    wikitext = wikitext.shuffle(seed=42)

from transformers import AutoTokenizer

context_length = 1024
tokenizer = AutoTokenizer.from_pretrained('gpt2', cache_dir=data_dir)

print(tokenizer.bos_token_id, tokenizer.eos_token_id)

# %%
import random

def tokenize(element):
    outputs = tokenizer(
        element["text"],
        truncation=False)
    
    input_batch = []
    for input_ids in outputs["input_ids"]:
        input_batch.append(tokenizer.bos_token_id)
        input_batch.extend(input_ids)
        input_batch.append(tokenizer.eos_token_id)
    
    factored = []
    for i in range(len(input_batch) // context_length):
        factored.append(input_batch[i * context_length:(i+1) * context_length])

    return {"input_ids": factored}


tokenized_datasets = wikitext.map(
    tokenize, batched=True, remove_columns='text'
)
print(tokenized_datasets)
print()

# %%
from transformers import DataCollatorForLanguageModeling

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)


# %%


from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig

config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    n_layer=6,
    n_embd=512,
    n_head=8,
)

print(config)
# %%
model = GPT2LMHeadModel(config)
model_size = sum(t.numel() for t in model.parameters())
print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")

# %%
for n, p in model.named_parameters():
    print(n, p.shape)

# %%
import numpy as np
from transformers import Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback

def compute_metrics(x):
    pred = x.predictions
    label = x.label_ids
    return {'accuracy': (np.argmax(pred, -1) == label).mean()}


args = TrainingArguments(
    output_dir="experiments/hf-baseline-1.0",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="steps",
    eval_steps=1000000,
    logging_steps=300,
    gradient_accumulation_steps=16,
    num_train_epochs=50,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=5_000,
    fp16=True,
    push_to_hub=False,
)

import wandb

log_steps = 400
acc_sum = 0.0
step_count = 0

class CustomTrainer(Trainer):
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        MAX: Subclassed to compute training accuracy.

        How the loss is computed by Trainer. By default, all models return the loss in
        the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)

        # MAX: Start of new stuff.
        if "labels" in inputs:
            preds = outputs.logits.detach()
            acc = (
                (preds.argmax(axis=-1) == inputs["labels"]).float().mean()
            )
            global step_count, acc_sum
            acc_sum += acc
            step_count += 1
            if step_count % log_steps == 0:
                self.log({'accuracy': acc_sum / step_count})
                step_count = 0

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of
            # ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics

)

# %%
trainer.train()

# %%
from torch.utils.data.dataloader import DataLoader

from torch.nn import CrossEntropyLoss
import torch


def keytoken_weighted_loss(inputs, logits):
    # Shift so that tokens < n predict n
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    # Calculate per-token loss
    loss_fct = CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    return loss


tokenized_datasets.set_format("torch")
train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=4, shuffle=True)
eval_dataloader = DataLoader(tokenized_datasets["test"], batch_size=4)


weight_decay = 0.1


def get_grouped_params(model, no_decay=["bias", "LayerNorm.weight"]):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]


def evaluate():
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(batch["input_ids"], labels=batch["input_ids"])

        losses.append(outputs.loss)
    loss = torch.mean(torch.cat(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return loss.item(), perplexity.item()

from torch.optim import AdamW

optimizer = AdamW(get_grouped_params(model), lr=5e-4)

from accelerate import Accelerator

accelerator = Accelerator(fp16=True)

model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

from transformers import get_scheduler

num_train_epochs = 1
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch
print("n_train_steps", num_training_steps)
print("num_training_steps")
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=1_000,
    num_training_steps=num_training_steps,
)

# %%
from tqdm.notebook import tqdm
import wandb

run = wandb.init(project='huggingface', entity='allanl', dir='experiments/hf-baseline-1.1',
        id='hf-baseline-1.2', resume='allow')

gradient_accumulation_steps = 8
eval_steps = 5_000

avg_metrics = {'train/loss': 0.0, 'train/accuracy': 0.0}
steps = 0
model.train()

completed_steps = 0
num_train_epochs = 5
for epoch in range(num_train_epochs):
    for step, batch in tqdm(
        enumerate(train_dataloader, start=1), total=num_training_steps
    ):
        logits = model(batch["input_ids"]).logits
        loss = keytoken_weighted_loss(batch["input_ids"], logits)
        avg_metrics['train/loss'] += float(loss)
        avg_metrics['train/accuracy'] += float((batch['input_ids'][..., 1:] == logits[..., :-1, :].argmax(-1)).float().mean().item())
        steps += 1
        if step % 100 == 0:
            for k in avg_metrics: avg_metrics[k] /= steps
            print(avg_metrics)
            wandb.log(avg_metrics)
            for k in avg_metrics: avg_metrics[k] = 0.0
            steps = 0

        loss = loss / gradient_accumulation_steps
        accelerator.backward(loss)
        if step % gradient_accumulation_steps == 0:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            completed_steps += 1
        