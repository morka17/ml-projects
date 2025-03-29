import torch
import torch.nn as nn
from typing import Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BigramLanguageModel(nn.Module):
    """A simple bigram language model for character-level text generation.
    
    This model uses an embedding layer followed by a linear layer to predict
    the next character in a sequence.
    
    Args:
        vocab_size (int): Size of the vocabulary
        n_embd (int): Size of the embedding vectors
        context_size (int, optional): Size of the context window. Defaults to 3
        
    Attributes:
        token_embedding_table (nn.Embedding): Embedding layer for input tokens
        lm_head (nn.Linear): Linear layer for final predictions
    """
    
    def __init__(self, vocab_size: int, n_embd: int, context_size: int = 3) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.context_size = context_size
        
        # Initialize layers
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(context_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the weights of the model.
        
        Args:
            module (nn.Module): The module to initialize
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of the model.
        
        Args:
            idx (torch.Tensor): Input token indices of shape (B, T)
            targets (torch.Tensor, optional): Target token indices of shape (B, T)
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Logits and loss if targets provided
        """
        B, T = idx.shape
        
        # Get embeddings
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        
        # Get predictions
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = nn.functional.cross_entropy(logits, targets)
            
        return logits, loss
    
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Generate new tokens given a context.
        
        Args:
            idx (torch.Tensor): Context tokens of shape (B, T)
            max_new_tokens (int): Number of new tokens to generate
            
        Returns:
            torch.Tensor: Generated tokens including context
        """
        for _ in range(max_new_tokens):
            # Get predictions
            logits, _ = self(idx)
            # Focus only on the last time step
            logits = logits[:, -1, :]  # (B, C)
            # Apply softmax to get probabilities
            probs = nn.functional.softmax(logits, dim=-1)  # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx 