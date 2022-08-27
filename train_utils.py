import os
import pathlib
from typing import List, Tuple

import ray
from tqdm import tqdm
import torch
from torch import nn

import wandb


def log_scalars(log: dict, step: int, prefix: str):
    prefix_copy = {f'{prefix}/{k}': v for k, v in log.items()}
    wandb.log(prefix_copy, step=step)

def remove_extra_checkpoints(folder='.', keep=5):
    past_checkpoints = get_checkpoints(folder)
    for i in range(0, max(len(past_checkpoints) - keep, 0)):
        os.remove(past_checkpoints[i])

def get_checkpoints(folder='.'):
    x = sorted(pathlib.Path(folder).glob('**/*.tar'))
    if x is None:
        return []
    x = [str(i) for i in x]
    x = [int(os.path.basename(i)[:-4]) for i in x]  # sort by int
    x.sort() # latest checkpoint comes last
    x = [os.path.join(folder, f'{str(i)}.tar') for i in x]
    return x

def save_checkpoint(state: dict, step: int, folder='.'):
    torch.save(state, os.path.join(folder, str(step) + ".tar"))

def call_functions(tensor, fn):
    """Helper function for recursive_cast"""
    return fn(tensor)

def recursive_cast(nested_tensor, fn):
    """Recursively casts nested tensor lists by attr_args, returns new casted results"""
    if isinstance(nested_tensor, list) or isinstance(nested_tensor, tuple):
        nest = []
        for i in range(len(nested_tensor)):
            nest.append(recursive_cast(nested_tensor[i], fn))
        return nest
    elif isinstance(nested_tensor, dict):
        nest = {}
        for k in nested_tensor:
            nest[k] = recursive_cast(nested_tensor[k], fn)
        return nest
    return fn(nested_tensor)


@ray.remote
class Logger:
    def __init__(self, init_fn, metrics) -> None:
        self.run = init_fn()
        self.metrics = metrics
        self.log_loss = {}
    
    def step(self, pred, y, loss_dict):
        self.metrics.update(pred, y)
        for k in loss_dict:
            if k in self.log_loss:
                self.log_loss[k] += float(loss_dict[k])
            else:
                self.log_loss[k] = float(loss_dict[k])
    
    def log(self, steps, steps_since_log, prefix='train'):
        # accumulated greater than the log_metric_step threshold
        # this step is very slow, move to another thread
        log_dict = {}
        log_dict.update({k: v / steps_since_log for k, v in self.log_loss.items()})
        log_dict.update(self.metrics.compute())
        log_scalars(log_dict, steps, prefix)
        self.metrics.reset()
        for k in self.log_loss:
            self.log_loss[k] = 0.0


def load_checkpoint(
    run_dir,
    model,
    criterion=None,
    optimizer=None,
    lr_scheduler=None
):
    try:
        checkpoints = get_checkpoints(run_dir)
        print("loading from saved checkpoint:", checkpoints[-1])
        state_dict = torch.load(checkpoints[-1])
        model.load_state_dict(state_dict['model'])

        if optimizer is not None: optimizer.load_state_dict(state_dict['optmizer'])
        if criterion is not None: criterion.load_state_dict(state_dict['criterion'])
        if lr_scheduler is not None: lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
    except Exception as e:
        print("unable to load", e)


def training_loop(
    model,
    train_set,
    criterion,
    optimizer,
    lr_scheduler,
    logger,
    log_scalar_metric_step,
    checkpoint_step,
    epochs,
    batch_size,
    eval_set=None,
    eval_logger=None,
    run_dir = ".",
    accumulate_gradient_step=1,
    grad_clip_norm=0.0,
    mixed_precision=False
):
    """
    Expects:
        train_set:
            must have member get() -> Tuple[x: Any, y: Any]
        model: 
            inheriting torch.nn.Module
            overloading __call__(x) -> pred: Any
            where x is the same as returned by train_set.get()
        criterion:
            inheriting torch.nn.Module
            overloading __call__(pred, y) -> Tuple[total_loss: torch.Tensor, loss_dict: Dict[str, float]]
        optimizer:
            standard torch.optim optimizer
        logger:
            must have member step.remote(pred, y, loss_dict) -> None
            and member log.remote() -> None
    """
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    
    scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)

    # load checkpoint, if any
    if os.path.exists(run_dir):
        try:
            checkpoints = get_checkpoints(run_dir)
            print("loading from saved checkpoint:", checkpoints[-1])
            state_dict = torch.load(checkpoints[-1])
            model.load_state_dict(state_dict['model'])
            optimizer.load_state_dict(state_dict['optmizer'])
            scaler.load_state_dict(state_dict['grad_scaler'])
            criterion.load_state_dict(state_dict['criterion'])
            lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
            steps = state_dict['steps']
        except Exception as e:
            print("unable to load due to:", e)

    exec_steps = 0
    steps = 0
    samples_shown = batch_size * accumulate_gradient_step
    dataset_len = None
    
    optimizer.zero_grad()

    for i in range(epochs):
        print('epoch number', i+1)
        if hasattr(train_set, "__len__"):
            dataset_len = len(dataset_len)
        elif i == 1:
            dataset_len = exec_steps
        
        for data in tqdm(train_set, total=dataset_len):
            exec_steps += 1
            gpu_data = recursive_cast(data, lambda x: torch.from_numpy(x).cuda())
            
            x, y = gpu_data
            with torch.cuda.amp.autocast(mixed_precision):
                model.train()
                pred = model(x)
                total_loss, loss_dict = criterion(pred, y)
                total_loss_ = total_loss / accumulate_gradient_step
                assert not torch.isnan(total_loss_).any()
            
            scaler.scale(total_loss_).backward()
            
            if exec_steps % accumulate_gradient_step == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()


                # commence logging and checkpointing operations
                remove_extra_checkpoints(run_dir, keep=3)
                
                steps += 1

                to_cpu = lambda x: recursive_cast(x, lambda x: x.argmax(-1).cpu().numpy())
                logger.step.remote(to_cpu(pred), data[1], recursive_cast(loss_dict, lambda x: float(x.detach().cpu())))

                if eval_set is not None and eval_logger is not None:
                    with torch.no_grad():
                        model.eval()
                        with torch.cuda.amp.autocast(mixed_precision):
                            eval_data = eval_set.get()
                            gpu_data = recursive_cast(eval_data, lambda x: torch.from_numpy(x).cuda())
                            x, y = gpu_data
                            pred = model(x)
                            # don't compute eval loss for now
                            eval_logger.step.remote(to_cpu(pred), torch.from_numpy(eval_data[1]), {})
                
                
                if steps % log_scalar_metric_step == 0: # accumulated greater than the log_metric_step threshold
                    wandb.log({}) # dummy log for wandb to log gradients
                    # stretch by samples_shown, to make log invariant to batch_size
                    # effectively logging sample efficiency as well
                    logger.log.remote(steps * samples_shown, log_scalar_metric_step)
                    if eval_logger is not None:
                        eval_logger.log.remote(steps * samples_shown, log_scalar_metric_step, prefix='eval')
                
                if steps % checkpoint_step == 0:
                    state_dict = {}
                    state_dict['model'] = model.state_dict()
                    state_dict['optmizer'] = optimizer.state_dict()
                    state_dict['grad_scaler'] = scaler.state_dict()
                    state_dict['criterion'] = criterion.state_dict()
                    state_dict['lr_scheduler'] = lr_scheduler.state_dict()
                    state_dict['steps'] = steps
                    save_checkpoint(state_dict, steps * samples_shown, run_dir)

        state_dict = {}
        state_dict['model'] = model.state_dict()
        state_dict['optmizer'] = optimizer.state_dict()
        state_dict['grad_scaler'] = scaler.state_dict()
        state_dict['criterion'] = criterion.state_dict()
        state_dict['lr_scheduler'] = lr_scheduler.state_dict()
        state_dict['steps'] = steps
        save_checkpoint(state_dict, steps * samples_shown, run_dir)
