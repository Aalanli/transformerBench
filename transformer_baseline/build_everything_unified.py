# %%
import torch
import wandb
from misc import EasyDict
import ray


################################################
# build the dataset and data transformations
# since Dataset is generic w.r.t the maestro 
# dataset, but transforms may be model dependent
################################################
from data.build_dataset import build_bert, iter_and_batch_multiprocess
from itertools import cycle

d_args = EasyDict()

d_args.seq_len               = 1025               # one extra token since loss is auto-regressive
d_args.batch_size            = 8

train_dataset_fn = build_bert
train_dataset = iter_and_batch_multiprocess(train_dataset_fn, d_args.batch_size, d_args.seq_len, buffer_size=4)
train_dataset = cycle(train_dataset)
eval_dataset  = None

###############################################################
# build model first, since optimizers and lr will need it later
###############################################################
from transformer_baseline.alibi_unified import build_model_and_criterion, Metrics

m_args = EasyDict()

m_args.vocab                 = 50257
m_args.embed_dim             = 512
m_args.n_layers              = 6
m_args.n_heads               = 8
m_args.max_sequence          = 1024
m_args.proj_forward          = 4096
m_args.dropout               = 0.1 
m_args.activation            = 'gelu'


model, criterion = build_model_and_criterion(m_args)
train_metrics = Metrics()
eval_metrics = None

##################################
# simple optimizer hyperparameters
##################################
opt_args = EasyDict()
opt_args.lr                = 1e-3
opt_args.weight_decay      = 1e-4

optimizer = torch.optim.AdamW(model.parameters(), **opt_args)

opt_args.lr_drop           = 50
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt_args.lr_drop)



# %%
###################
# commence training
###################
if __name__ == '__main__':
    import os
    from train_utils import training_loop, Logger
    run_name = 'baseline_alibi-1.22'
    train_args = EasyDict()
    train_args.epochs                     = 120
    train_args.run_dir                    = f'experiments/{run_name}'
    train_args.log_scalar_metric_step     = 100 * d_args.batch_size
    train_args.checkpoint_step            = 500 * d_args.batch_size
    train_args.batch_size                 = d_args.batch_size
    train_args.mixed_precision            = True


    configs = EasyDict()
    configs.model_args                    = m_args
    configs.optimizer_args                = opt_args
    configs.data_args                     = d_args
    configs.batch_size                    = d_args.batch_size
    configs.description                   = 'Baseline alibi with no modifications'
    configs.dataset                       = 'Bert'

    if not os.path.exists(train_args.run_dir):
        os.makedirs(train_args.run_dir)
    
    run_fn = lambda: wandb.init(project='TransformerBench', entity='allanl', dir=train_args.run_dir, 
        group=run_name, id=run_name, config=configs, resume='allow')

    #run_fn = lambda: None
    train_logger = Logger.remote(run_fn, train_metrics)
    eval_logger = None  # Logger.remote(run_fn, eval_metrics)

    model = model.cuda()
    criterion = criterion.cuda()

    run = run_fn()
    with run:
        wandb.watch(model, log_freq=500)
        training_loop(model, train_dataset, criterion, optimizer, lr_scheduler, train_logger, eval_logger=eval_logger, eval_set=eval_dataset, **train_args)
