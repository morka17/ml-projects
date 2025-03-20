import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Set, Tuple
import numpy as np
from tqdm import tqdm
from collections import Counter
import json
import os
from training_data import TrainingData

class TokenizerTrainer:
    def __init__(self, tokenizer, vocab_size: int = 50257):
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_data = TrainingData()
        
    def train_on_texts(self, texts: List[str], epochs: int = 5, batch_size: int = 32):
        """Train the tokenizer on a list of texts."""
        # Collect byte pairs from all texts
        byte_pairs = self._collect_byte_pairs(texts)
        
        # Initialize embeddings for byte pairs
        self._initialize_embeddings(byte_pairs)
        
        # Training loop
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            self._train_epoch(texts, batch_size)
            
        # Update tokenizer vocabulary with learned tokens
        self._update_tokenizer_vocab()
        
    def train_on_dataset(self, dataset_name: str = 'mixed', epochs: int = 5, batch_size: int = 32):
        """Train the tokenizer on a predefined dataset."""
        texts = self.training_data.get_dataset(dataset_name)
        return self.train_on_texts(texts, epochs, batch_size)
    
    def train_on_all_datasets(self, epochs: int = 5, batch_size: int = 32):
        """Train the tokenizer on all available datasets."""
        all_texts = []
        for dataset_name in self.training_data.datasets:
            all_texts.extend(self.training_data.get_dataset(dataset_name))
        return self.train_on_texts(all_texts, epochs, batch_size)
    
    def _collect_byte_pairs(self, texts: List[str]) -> Set[Tuple[bytes, bytes]]:
        """Collect all byte pairs from the texts."""
        byte_pairs = set()
        for text in texts:
            text_bytes = text.encode('utf-8')
            for i in range(len(text_bytes) - 1):
                byte_pairs.add((text_bytes[i:i+1], text_bytes[i+1:i+2]))
        return byte_pairs
    
    def _initialize_embeddings(self, byte_pairs: Set[Tuple[bytes, bytes]]):
        """Initialize embeddings for byte pairs."""
        self.byte_pair_embeddings = nn.Embedding(len(byte_pairs), 256)
        self.byte_pair_to_idx = {pair: idx for idx, pair in enumerate(byte_pairs)}
        self.idx_to_byte_pair = {idx: pair for pair, idx in self.byte_pair_to_idx.items()}
        
    def _train_epoch(self, texts: List[str], batch_size: int):
        """Train for one epoch."""
        optimizer = optim.Adam(self.byte_pair_embeddings.parameters())
        criterion = nn.CrossEntropyLoss()
        
        # Create batches
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        
        for batch in tqdm(batches, desc="Training"):
            optimizer.zero_grad()
            
            # Process batch
            batch_loss = 0
            for text in batch:
                text_bytes = text.encode('utf-8')
                for i in range(len(text_bytes) - 1):
                    pair = (text_bytes[i:i+1], text_bytes[i+1:i+2])
                    if pair in self.byte_pair_to_idx:
                        pair_idx = self.byte_pair_to_idx[pair]
                        next_byte = text_bytes[i+2:i+3] if i + 2 < len(text_bytes) else b'\x00'
                        
                        # Get embedding and predict next byte
                        embedding = self.byte_pair_embeddings(torch.tensor([pair_idx], device=self.device))
                        prediction = torch.matmul(embedding, self.byte_pair_embeddings.weight.t())
                        
                        # Calculate loss
                        target = torch.tensor([next_byte[0]], device=self.device)
                        loss = criterion(prediction, target)
                        batch_loss += loss
            
            # Backpropagate
            batch_loss.backward()
            optimizer.step()
    
    def _update_tokenizer_vocab(self):
        """Update the tokenizer's vocabulary with learned tokens."""
        # Sort byte pairs by frequency
        byte_pair_freqs = Counter()
        for text in self.tokenizer.training_texts:
            text_bytes = text.encode('utf-8')
            for i in range(len(text_bytes) - 1):
                pair = (text_bytes[i:i+1], text_bytes[i+1:i+2])
                if pair in self.byte_pair_to_idx:
                    byte_pair_freqs[pair] += 1
        
        # Add most frequent byte pairs to vocabulary
        sorted_pairs = sorted(byte_pair_freqs.items(), key=lambda x: x[1], reverse=True)
        for (pair, _) in sorted_pairs:
            if len(self.tokenizer.vocab) >= self.vocab_size:
                break
            merged = b''.join(pair)
            if merged not in self.tokenizer.vocab:
                self.tokenizer.vocab[merged] = len(self.tokenizer.vocab)
    
    def save_model(self, path: str):
        """Save the trained model."""
        state = {
            'byte_pair_embeddings': self.byte_pair_embeddings.state_dict(),
            'byte_pair_to_idx': self.byte_pair_to_idx,
            'idx_to_byte_pair': self.idx_to_byte_pair,
            'vocab': self.tokenizer.vocab
        }
        torch.save(state, path)
    
    def load_model(self, path: str):
        """Load a trained model."""
        if os.path.exists(path):
            state = torch.load(path)
            self.byte_pair_embeddings.load_state_dict(state['byte_pair_embeddings'])
            self.byte_pair_to_idx = state['byte_pair_to_idx']
            self.idx_to_byte_pair = state['idx_to_byte_pair']
            self.tokenizer.vocab = state['vocab']
            return True
        return False 