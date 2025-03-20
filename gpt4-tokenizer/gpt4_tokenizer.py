import re
from typing import List, Dict, Set, Tuple
import unicodedata
import torch
import numpy as np
import os

class GPT4Tokenizer:
    def __init__(self):
        # Special tokens
        self.special_tokens = {
            '<|endoftext|>': 50256,
            '<|im_start|>': 50257,
            '<|im_end|>': 50258,
            '<|system|>': 50259,
            '<|user|>': 50260,
            '<|assistant|>': 50261,
            '<|unknown|>': 50262,  # New special token for unknown tokens
        }
        
        # Initialize the vocabulary with bytes
        self.vocab = {}
        self.merges = {}
        self.training_texts = []  # Store training texts
        self._initialize_vocab()
        
    def _initialize_vocab(self):
        """Initialize the vocabulary with bytes and special tokens."""
        # Add bytes to vocabulary
        for i in range(256):
            self.vocab[bytes([i])] = i
            
        # Add special tokens to vocabulary
        for token, idx in self.special_tokens.items():
            self.vocab[token.encode('utf-8')] = idx
            
    def _get_byte_pairs(self, word: bytes) -> Set[Tuple[bytes, bytes]]:
        """Get all consecutive pairs of bytes in a word."""
        pairs = set()
        for i in range(len(word) - 1):
            pairs.add((word[i:i+1], word[i+1:i+2]))
        return pairs
    
    def _merge_byte_pairs(self, word: bytes, pair: Tuple[bytes, bytes]) -> bytes:
        """Merge a pair of bytes in a word."""
        merged = b''
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i:i+1] == pair[0] and word[i+1:i+2] == pair[1]:
                merged += pair[0] + pair[1]
                i += 2
            else:
                merged += word[i:i+1]
                i += 1
        return merged
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text by converting to NFKC form and handling whitespace."""
        # Convert to NFKC form
        text = unicodedata.normalize('NFKC', text)
        
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Handle special whitespace characters
        text = text.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
        
        return text
    
    def encode(self, text: str) -> List[int]:
        """Encode text into tokens."""
        # Normalize text
        text = self._normalize_text(text)
        
        # Convert text to bytes
        text_bytes = text.encode('utf-8')
        
        # Initialize tokens list
        tokens = []
        
        # Process text in chunks
        i = 0
        while i < len(text_bytes):
            # Try to match special tokens first
            matched = False
            for token, idx in self.special_tokens.items():
                token_bytes = token.encode('utf-8')
                if text_bytes.startswith(token_bytes, i):
                    tokens.append(idx)
                    i += len(token_bytes)
                    matched = True
                    break
            
            if not matched:
                # Try to match longest possible token
                longest_match = None
                longest_length = 0
                
                for token, idx in self.vocab.items():
                    if text_bytes.startswith(token, i):
                        if len(token) > longest_length:
                            longest_match = idx
                            longest_length = len(token)
                
                if longest_match is not None:
                    tokens.append(longest_match)
                    i += longest_length
                else:
                    # Handle unknown token
                    tokens.append(self.special_tokens['<|unknown|>'])
                    i += 1
        
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        """Decode tokens back into text."""
        # Create reverse vocabulary mapping
        reverse_vocab = {idx: token for token, idx in self.vocab.items()}
        
        # Decode tokens
        text_bytes = b''
        for token in tokens:
            if token in reverse_vocab:
                text_bytes += reverse_vocab[token]
            else:
                # Handle unknown token
                text_bytes += b'<|unknown|>'
        
        # Convert bytes back to text
        return text_bytes.decode('utf-8', errors='replace')
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the text."""
        return len(self.encode(text))
    
    def train(self, texts: List[str], vocab_size: int = 50257, epochs: int = 5, batch_size: int = 32):
        """Train the tokenizer on a list of texts."""
        from tokenizer_trainer import TokenizerTrainer
        trainer = TokenizerTrainer(self, vocab_size)
        trainer.train_on_texts(texts, epochs, batch_size)
        return trainer
    
    def save(self, path: str):
        """Save the tokenizer state."""
        state = {
            'vocab': self.vocab,
            'special_tokens': self.special_tokens,
            'merges': self.merges
        }
        torch.save(state, path)
    
    def load(self, path: str):
        """Load the tokenizer state."""
        if os.path.exists(path):
            state = torch.load(path)
            self.vocab = state['vocab']
            self.special_tokens = state['special_tokens']
            self.merges = state['merges']
            return True
        return False 