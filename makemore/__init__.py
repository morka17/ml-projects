"""
makemore - A character-level language model for generating names.

This package provides functionality for training and using a character-level
language model to generate names. It includes dataset handling, model architecture,
and training utilities.
"""

from .dataset import NameDataset, create_dataloader, get_default_dataset
from .model import BigramLanguageModel
from .train import Trainer

__version__ = '0.1.0'
__author__ = 'Your Name'

__all__ = [
    'NameDataset',
    'create_dataloader',
    'get_default_dataset',
    'BigramLanguageModel',
    'Trainer',
] 