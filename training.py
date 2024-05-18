from math import ceil
import time
from datetime import datetime
import os 
import shutil
import argparse
from argparse import ArgumentParser

import torch
import pandas as pd
import yaml
import wandb

from data import get_data, ALL_OPERATIONS, get_vocab_size
from model import Transformer, create_model
from utils import param_stats, evaluate, get_run_name
from loss_contour import explore_gradient_directions, plot_loss_acc_contours_2D, plot_loss_acc_contours_3D

# model architecture
num_layers = 2
dim_model = 128
num_heads = 4
init_from = "scratch"
compile = False

# train 
steps = 0
eval_batches = 2
batch_size = 512
max_steps = 80_000
device = "cuda"
train_frac = 0.5
learning_rate = 1e-3
weight_decay = 1
seed = 42
# data
prime = 97
operation = "xy/y"
context_size = 4 
# log
log_interval = 50
wandb_log = False
save_checkpoint = False
save_log = False
print_iter = False
time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
date_str = datetime.now().strftime("%Y-%m-%d")
val_acc_extra_steps=1_000_000  # not using right now
ckpt_name = 'ckpt'
ckpt_interval = 5000
ckpt_max_steps = 80_000

# configurations
config_file = ''

# config 
# we want a single global dict that functions can use / update
# we cannot update by calling globals()
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys}

def train_batch(model, train_loader, optimizer, scheduler):
    """
    training for a single batch
    """
    global config

    # Set model to training mode
    model.train()
    criterion = torch.nn.CrossEntropyLoss()

    batch = next(iter(train_loader))
    vocab_size = get_vocab_size(prime)
    inputs, labels = batch
    assert list(inputs.size()) == [batch_size, context_size]
    assert list(labels.size()) == [batch_size]

    if config['device'] == 'cuda':
        inputs = inputs.pin_memory().to(config['device'], non_blocking=True)
        labels = labels.pin_memory().to(config['device'], non_blocking=True)

    optimizer.zero_grad()
    
    # forward pass
    model_output = model(inputs)
    assert list(model_output.size()) == [context_size, batch_size, vocab_size]
    output = model_output[-1,:,:]
    assert list(output.size()) == [batch_size, vocab_size]
    

    loss = criterion(output, labels)
    # because we set token values to be same as the original numbers 
    # we are matching argmax index with label
    acc = (torch.argmax(output, dim=1) == labels).sum() / len(labels)
    
    # backward pass
    loss.backward()

    # calculating weight and gradient stats
    # generally done after backward pass and before optimizer.step()
    weight_stats, grad_stats = param_stats(model)

    optimizer.step()
    scheduler.step()

    return acc, loss, weight_stats, grad_stats

def run_train_loop():
    global config
    
    torch.manual_seed(config["seed"])
    if config['device'] == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Log config 
    run_name = get_run_name(config)
    log_prefix=f"log/{date_str}/{run_name}_{time_str}"

    if config['save_log']:    
        config_df = pd.DataFrame([config.values()], columns=config.keys())
        os.makedirs("log", exist_ok=True)
        os.makedirs("log/{}".format(date_str), exist_ok=True)
        config_df.to_csv(f"{log_prefix}_config.csv", index=False)
    
    if config['wandb_log']:
        wandb.init(project="grokking", 
                name=f"{run_name}_{time_str}", 
                config=config)
    
    if config['save_checkpoint']:
        os.makedirs("checkpoints", exist_ok=True)
    
    # Create model
    model, optimizer, scheduler, config_updt = create_model(config)
    config.update(config_updt)
    print("config\n", config)

    # Generate data
    train_loader, val_loader = get_data(config['operation'], config['prime'], config['train_frac'], config['batch_size'])

    # Set log fields
    train_iter_columns_fixed= ["step", "train_acc", "train_loss"]
    param_fields = [
        "attn_0_in_proj_weight", "attn_0_out_proj_weight",
        "attn_1_in_proj_weight", "attn_1_out_proj_weight",
    ]
    train_iter_columns_additional= ( [ f"param_mean_{field}" for field in param_fields ] 
        + [ f"param_norm_{field}" for field in param_fields ]
        + [ f"grad_mean_{field}" for field in param_fields ]
        + [ f"grad_norm_{field}" for field in param_fields ] 
    )
    
    train_iter_df = pd.DataFrame(columns=train_iter_columns_fixed) # + train_iter_columns_additional
    val_iter_df = pd.DataFrame(columns=["step", "val_acc", "val_loss"])

    # Run iterations
    steps = config['steps']
    run_start = time.time()
    val_acc_95_seen = 0
    while steps <= config['max_steps']:
        # torch.manual_seed(config["seed"] + steps)

        # Save checkpoint
        if steps <= ckpt_max_steps and steps % config['ckpt_interval'] == 0 \
            and config['save_checkpoint']:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'steps': steps,
                'config': config
            }
            torch.save(checkpoint, f"checkpoints/{get_run_name(config, with_steps=True)}_{time_str}.pt")
        
        iter_start = time.time()
        iter_acc, iter_loss, weight_stats, grad_stats \
            = train_batch(model, train_loader, optimizer, scheduler)
        iter_time = time.time() - iter_start

        steps += 1
        config['steps'] = steps

        # Log
        if steps % config['log_interval'] == 0:
            train_acc, train_loss, val_acc, val_loss = evaluate(model, config, train_loader, val_loader)

            if print_iter:
                print(f"iter_acc: {iter_acc:.2f}, iter_loss: {iter_loss:.4f}, iter_time: {iter_time:.2f}")
            print(f"step {steps}: train_acc: {train_acc:.2f}, train_loss: {train_loss:.4f}, val_acc: {val_acc:.2f}, val_loss: {val_loss:.4f}")

            # write 2 copies to avoid data loss at Ctrl+C
            if config['save_log']:
                train_data_fixed = [steps, train_acc.cpu().detach().numpy().item(), 
                                       train_loss.cpu().detach().numpy().item()]
                train_data_additional = [] 
                param_names = ["model.0.self_attn.in_proj_weight", 
                             "model.0.self_attn.out_proj.weight", 
                             "model.1.self_attn.in_proj_weight",
                             "model.1.self_attn.out_proj.weight"]
                for name in param_names:
                    train_data_additional.append(weight_stats[f"param/mean/{name}"])
                
                for name in param_names: 
                    train_data_additional.append(weight_stats[f"param/norm/{name}"])
                
                for name in param_names:
                    train_data_additional.append(grad_stats[f"grad/mean/{name}"])
                
                for name in param_names: 
                    train_data_additional.append(grad_stats[f"grad/norm/{name}"])

                train_data_df = pd.DataFrame([train_data_fixed], # + train_data_additional 
                                            columns=train_iter_df.columns)
                train_iter_df = pd.concat([train_iter_df, train_data_df], ignore_index=True) \
                                if len(train_iter_df) > 0 else train_data_df

                val_data_df = pd.DataFrame([[steps, val_acc.cpu().detach().numpy().item(), 
                                        val_loss.cpu().detach().numpy().item()]], columns=val_iter_df.columns)
                val_iter_df = pd.concat([val_iter_df, val_data_df], ignore_index=True) if len(val_iter_df) > 0 else val_data_df

                train_iter_df.to_csv(f"{log_prefix}_train.csv", index=False)
                val_iter_df.to_csv(f"{log_prefix}_val.csv", index=False)

                shutil.copy(f"{log_prefix}_train.csv", f"{log_prefix}_train_bk.csv")
                shutil.copy(f"{log_prefix}_val.csv", f"{log_prefix}_val_bk.csv")

            if config['wandb_log']:
                wandb.log({
                    "iter/acc": iter_acc,
                    "iter/loss": iter_loss,
                    "train/acc": train_acc,
                    "train/loss": train_loss,
                    "val/acc": val_acc,
                    "val/loss": val_loss,
                    "total_time": time.time() - run_start,
                    **weight_stats,
                    **grad_stats
                })

            if val_acc > 0.95:
                val_acc_95_seen += 1

        if val_acc_95_seen * log_interval >= max(val_acc_extra_steps, steps // 10):
            break

# steps = epochs * len(train_loader)
# train_size = prime * prime * train_frac
# len(train_loader) = ceil(train_size / batch_size)

def parse_args():
    parser = ArgumentParser()
    # model architecture
    parser.add_argument("--num_layers", type=int, default=num_layers)
    parser.add_argument("--dim_model", type=int, default=dim_model)
    parser.add_argument("--num_heads", type=int, default=num_heads)
    parser.add_argument("--init_from", type=str, default=init_from)
    
    # train
    parser.add_argument("--eval_batches", type=int, default=eval_batches)
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--max_steps", type=int, default=max_steps)
    parser.add_argument("--device", type=str, default=device)
    parser.add_argument("--train_frac", type=float, default=train_frac)
    parser.add_argument("--learning_rate", type=float, default=learning_rate)
    parser.add_argument("--weight_decay", type=float, default=weight_decay)
    parser.add_argument("--seed", type=int, default=seed)
    
    # data
    parser.add_argument("--prime", type=int, default=prime)
    parser.add_argument("--operation", type=str, choices=ALL_OPERATIONS.keys(), default=operation)
    
    # log
    # for bool args, do not use to set to False, --arg False will not work
    parser.add_argument("--log_interval", type=int, default=log_interval)
    parser.add_argument("--wandb_log", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--save_checkpoint", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--save_log", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--ckpt_name", type=str, default=ckpt_name)
    parser.add_argument("--ckpt_interval", type=int, default=ckpt_interval)
    parser.add_argument("--ckpt_max_steps", type=int, default=ckpt_max_steps)
    
    #configurations
    parser.add_argument("--config_file", type=str, default=config_file)
    args = vars(parser.parse_args())

    # Does not work because we have default values in args. 
    # No way to distinguish given vs default. 
    # if args["config_file"]:
    #     with open(args["config_file"], "r") as f:
    #         config_yaml = yaml.safe_load(f)
    #     config_yaml.update(args)
    #     args = config_yaml

    return args

def set_config(k, v):
    global config
    if k in config:
        config[k] = v

def plot_loss_acc_contours():
    global config
    torch.manual_seed(config["seed"])

    model, _, _, config_updt = create_model(config)
    config.update(config_updt)
    train_loader, val_loader = get_data(config['operation'], config['prime'], config['train_frac'], config['batch_size'])
    train_acc, train_loss, val_acc, val_loss = evaluate(model, config, train_loader, val_loader)
    print("train_acc", train_acc, "train_loss", train_loss, "val_acc", val_acc, "val_loss", val_loss)

    results = explore_gradient_directions(model, config, train_loader, val_loader, steps=21, search_range=1)
    plot_loss_acc_contours_2D(results, f"figures/{get_run_name(config, with_steps=True)}")
    plot_loss_acc_contours_3D(results, f"figures/{get_run_name(config, with_steps=True)}")

if __name__ == "__main__":
    # Config priority (from lowest to highest):
    # 1. Default global values
    # 2. Command line arguments

    args = parse_args()

    for k in args:
        if args[k] is not None:
            set_config(k, args[k])

    run_train_loop()
    # plot_loss_acc_contours()
