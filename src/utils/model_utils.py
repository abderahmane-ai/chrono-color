import torch
import torch.nn as nn

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_summary(model, name):
    """Print a summary of the model"""
    print(f"\n{name}:")
    print(f"Total parameters: {count_parameters(model):,}")
    print(model)

def freeze_model(model):
    """Freeze all parameters in a model"""
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_model(model):
    """Unfreeze all parameters in a model"""
    for param in model.parameters():
        param.requires_grad = True