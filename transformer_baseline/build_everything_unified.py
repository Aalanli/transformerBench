# %%
import torch
import wandb
from misc import EasyDict


################################################
# build the dataset and data transformations
# since Dataset is generic w.r.t the maestro 
# dataset, but transforms may be model dependent
################################################
from data.build_dataset import build_bert, build_wiki

d_args = EasyDict()

d_args.seq_len               = 1025               # one extra token since loss is auto-regressive
d_args.batch_size            = 8

train_dataset = build_wiki(**d_args, buffer_size=4)
eval_dataset  = None

# %%
###############################################################
# build model first, since optimizers and lr will need it later
###############################################################
from transformer_baseline.alibi_unified import Criterion, Metrics
from transformer_baseline.simple_attnv2 import Transformer

m_args = EasyDict()

m_args.vocab                 = 50257
m_args.embed_dim             = 512
m_args.n_layers              = 6
m_args.n_heads               = 8
m_args.max_sequence          = 1024
m_args.proj_forward          = 2048
m_args.dropout               = 0.1 
m_args.activation            = 'relu'

model = Transformer(**m_args)
criterion = Criterion()
train_metrics = Metrics()
eval_metrics = None
"""from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig

config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=50257,
    n_ctx=1024,
    bos_token_id=50256,
    eos_token_id=50256,
    n_layer=6,
    n_embd=512,
    n_head=8,
)

model = GPT2LMHeadModel(config)

class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x).logits
model = Wrapper(model)"""
model_size = sum(t.numel() for t in model.parameters())
print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")
for n, p in model.named_parameters():
    print(n, p.shape)

# %%
##################################
# simple optimizer hyperparameters
##################################
import lr_schedulers as lrs
opt_args = EasyDict()
opt_args.lr                  = 5e-4

def get_grouped_params(model, no_decay=["bias", "LayerNorm.weight"]):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": 0.1},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]

optimizer = torch.optim.AdamW(get_grouped_params(model), **opt_args)

opt_args.warm_up_steps       = 1000

#lr_scheduler = lrs.InvSqrtScheduler(optimizer, m_args.embed_dim, opt_args.warm_up_steps)
lr_scheduler = lrs.get_linear_schedule_with_warmup(optimizer, 1000, 29416)


# %%
###################
# commence training
###################
if __name__ == '__main__':
    import os
    from train_utils import training_loop, Logger
    group_name = 'simple-baseline'
    run_name = '3.5'
    run_dir = group_name + "-" + run_name
    train_args = EasyDict()
    train_args.epochs                     = 3
    train_args.run_dir                    = f'experiments/{run_dir}'
    train_args.log_scalar_metric_step     = 70
    train_args.checkpoint_step            = 300
    train_args.mixed_precision            = True
    train_args.accumulate_gradient_step   = 4
    train_args.grad_clip_norm             = 1.0
    train_args.batch_size                 = d_args.batch_size

    configs = EasyDict()
    configs.model_args                    = m_args
    configs.optimizer_args                = opt_args
    configs.data_args                     = d_args
    configs.batch_size                    = d_args.batch_size
    configs.description                   = 'Baseline with no modifications'
    configs.dataset                       = 'wikitext'

    if not os.path.exists(train_args.run_dir):
        os.makedirs(train_args.run_dir)
    
    run_fn = lambda c: wandb.init(project='TransformerBench', entity='allanl', dir=train_args.run_dir, 
        group=group_name, id=run_name + c, config=configs, resume='allow')

    #run_fn = lambda: None
    train_logger = Logger.remote(lambda: run_fn('-l'), train_metrics)
    eval_logger = None  # Logger.remote(run_fn, eval_metrics)

    model = model.cuda()
    criterion = criterion.cuda()

    run = run_fn('-g')
    with run:
        wandb.watch(model, log_freq=50)
        training_loop(model, train_dataset, criterion, optimizer, lr_scheduler, train_logger, eval_logger=eval_logger, eval_set=eval_dataset, **train_args)

# %%

a = iter(train_dataset)

# %%
x = next(a)
y = model(torch.from_numpy(x[0]).cuda())

(y.argmax(-1) == torch.from_numpy(x[1]).cuda()).float().mean()