from functools import reduce 
import operator

import torch

def grad_norm_1(model):
    grad_norm = 0
    for p in model.parameters():
        grad_norm += p.grad.norm().item() ** 2
    grad_norm = grad_norm ** 0.5
    return grad_norm

def grad_norm_2(model):
    """
    clip_grad_norm_ clips the gradient norm (truncates at a max value)
    and returns the total norm of the parameter gradients

    Shouldn't it be returning the max if anytime the total norm exceeds the max?
    It was not the case in the result. Is that because it was totaling?
    """
    return torch.nn.utils.clip_grad_norm_(model.parameters(), 1000.0).item()

def layer_grad_norm(model, layer_name):
    """
    Gradient norm of a particular layer. 

    Example layer names: 'model.0', 'model.1.self_attn.out_proj' 
    """
    layer = dict(model.named_modules())[layer_name]
    grad_norm = 0
    for p in layer.parameters():
        grad_norm += p.grad.norm().item() ** 2
    grad_norm = grad_norm ** 0.5
    return grad_norm

def layer_grad_norms(model):
    """
    This function calculates the gradient norm for each layer (module) in the model.
    """
    grad_norms = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Module):
        # Check if the module has parameters and gradients
            if hasattr(module, 'parameters'):
                # Calculate the norm of the gradients for this layer's parameters
                param_grads = [p.grad for p in module.parameters() if p.grad is not None]
                if param_grads:
                    # Sum norms of individual parameter gradients for this layer
                    grad_norm = reduce(operator.add, [p.norm().item() ** 2 for p in param_grads]) ** 0.5
                    grad_norms[name] = grad_norm
    return grad_norms

def param_stats(model):
    """
    Returns dicts (weight_stats, grad_stats)
    """
    weight_stats = {}
    grad_stats = {}
    params_count = {}
    for name, param in model.named_parameters():
        param_filter = ('proj' in name) and ('weight' in name)
        if (param.grad is not None) and param_filter:
            params_count[name] = param.numel()
            weight_stats[f"param/mean/{name}"] = param.mean().item()
            weight_stats[f"param/std/{name}"] = param.std().item()
            weight_stats[f"param/norm/{name}"] = param.norm(2).item()
            
            grad_stats[f"grad/mean/{name}"] = param.grad.mean().item()
            grad_stats[f"grad/std/{name}"] = param.grad.std().item()
            grad_stats[f"grad/norm/{name}"] = param.grad.norm(2).item()

    return weight_stats, grad_stats     

def get_run_name(config, with_steps=False):
    run_name = f"{config['operation'].replace('/', '|')}" + \
        f"_f{str(config['train_frac'])}_p{config['prime']}_b{config['batch_size']}" + \
        f"_wd{config['weight_decay']}_sd{config['seed']}" + \
        (f"_st{config['steps']}" if with_steps else "")
    
    return run_name

def evaluate(model, config, train_loader, val_loader):
    """Measure accuracy and loss in train and validation sets"""
    # Set model to evaluation mode
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    for loader in [train_loader, val_loader]:
        correct = 0
        loss = 0.
        iterated = 0
        while iterated < config['eval_batches']:
            batch = next(iter(loader))    
            inputs, labels = batch
            if config['device'] == 'cuda':
                inputs = inputs.pin_memory().to(config['device'], non_blocking=True)
                labels = labels.pin_memory().to(config['device'], non_blocking=True)

            with torch.no_grad():
                output = model(inputs)[-1,:,:]
                correct += (torch.argmax(output, dim=1) == labels).sum()
                loss += criterion(output, labels) * len(labels)
            
            iterated += 1
        
        if loader == train_loader:
            train_acc = correct / (config['eval_batches'] * config['batch_size'])
            train_loss = loss / (config['eval_batches'] * config['batch_size'])
        else:
            val_acc = correct / (config['eval_batches'] * config['batch_size'])
            val_loss = loss / (config['eval_batches'] * config['batch_size'])

    return train_acc, train_loss, val_acc, val_loss