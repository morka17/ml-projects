import os
from typing import List, Dict
import json

class TrainingData:
    def __init__(self):
        self.datasets = {
            'code': self._load_code_dataset(),
            'text': self._load_text_dataset(),
            'mixed': self._load_mixed_dataset()
        }
    
    def _load_code_dataset(self) -> List[str]:
        """Load programming code samples."""
        return [
            "def hello_world():",
            "    print('Hello, World!')",
            "class Example:",
            "    def __init__(self):",
            "        self.value = 42",
            "    def get_value(self):",
            "        return self.value",
            "import torch",
            "import numpy as np",
            "def process_data(data):",
            "    return torch.tensor(data)",
            "for i in range(10):",
            "    if i % 2 == 0:",
            "        print(f'Even number: {i}')",
            "    else:",
            "        print(f'Odd number: {i}')"
        ]
    
    def _load_text_dataset(self) -> List[str]:
        """Load general text samples."""
        return [
            "The quick brown fox jumps over the lazy dog.",
            "To be, or not to be, that is the question.",
            "All that glitters is not gold.",
            "A journey of a thousand miles begins with a single step.",
            "The best way to predict the future is to create it.",
            "In three words I can sum up everything I've learned about life: it goes on.",
            "The only way to do great work is to love what you do.",
            "Life is what happens while you're busy making other plans.",
            "Success is not final, failure is not fatal: it is the courage to continue that counts.",
            "The future belongs to those who believe in the beauty of their dreams."
        ]
    
    def _load_mixed_dataset(self) -> List[str]:
        """Load mixed content samples."""
        return [
            "Here's a code example: def example(): return 42",
            "The API endpoint is: /api/v1/users",
            "Error code: 404 Not Found",
            "User ID: 12345",
            "Timestamp: 2024-03-20T10:30:00Z",
            "JSON response: {'status': 'success'}",
            "Database query: SELECT * FROM users",
            "Git commit: a1b2c3d4",
            "HTTP method: POST",
            "Content-Type: application/json"
        ]
    
    def get_dataset(self, name: str = 'mixed') -> List[str]:
        """Get a specific dataset by name."""
        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' not found. Available datasets: {list(self.datasets.keys())}")
        return self.datasets[name]
    
    def get_all_datasets(self) -> Dict[str, List[str]]:
        """Get all available datasets."""
        return self.datasets
    
    def save_dataset(self, name: str, data: List[str], path: str):
        """Save a dataset to a file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_dataset(self, path: str) -> List[str]:
        """Load a dataset from a file."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f) 