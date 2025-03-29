import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import json
from tqdm import tqdm

from .model import BigramLanguageModel
from .dataset import NameDataset, create_dataloader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Trainer:
    """Trainer class for the bigram language model.
    
    This class handles the training loop, evaluation, and model checkpointing.
    
    Args:
        model (BigramLanguageModel): The model to train
        train_dataset (NameDataset): Training dataset
        val_dataset (Optional[NameDataset]): Validation dataset
        learning_rate (float, optional): Learning rate. Defaults to 1e-3
        batch_size (int, optional): Batch size. Defaults to 32
        num_workers (int, optional): Number of dataloader workers. Defaults to 4
        device (str, optional): Device to train on. Defaults to 'cuda' if available
        checkpoint_dir (str, optional): Directory to save checkpoints. Defaults to 'checkpoints'
    """
    
    def __init__(
        self,
        model: BigramLanguageModel,
        train_dataset: NameDataset,
        val_dataset: Optional[NameDataset] = None,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        num_workers: int = 4,
        device: Optional[str] = None,
        checkpoint_dir: str = 'checkpoints'
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = Path(checkpoint_dir)
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Create dataloaders
        self.train_loader = create_dataloader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )
        self.val_loader = create_dataloader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        ) if val_dataset else None
        
        # Initialize optimizer
        self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        
    def save_checkpoint(self, epoch: int, loss: float) -> None:
        """Save a model checkpoint.
        
        Args:
            epoch (int): Current epoch
            loss (float): Current loss
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }
        path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, path)
        logger.info(f'Saved checkpoint to {path}')
        
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load a model checkpoint.
        
        Args:
            path (str): Path to the checkpoint file
            
        Returns:
            Dict[str, Any]: Checkpoint data
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint
        
    def train_epoch(self) -> float:
        """Train for one epoch.
        
        Returns:
            float: Average loss for the epoch
        """
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        with tqdm(self.train_loader, desc='Training') as pbar:
            for batch_idx, (x, y) in enumerate(pbar):
                # Move data to device
                x = x.to(self.device)
                y = y.to(self.device)
                
                # Forward pass
                logits, loss = self.model(x, y)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Update progress bar
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
                
        return total_loss / num_batches
        
    def evaluate(self) -> float:
        """Evaluate the model on the validation set.
        
        Returns:
            float: Average validation loss
        """
        if not self.val_loader:
            return 0.0
            
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            with tqdm(self.val_loader, desc='Evaluating') as pbar:
                for x, y in pbar:
                    # Move data to device
                    x = x.to(self.device)
                    y = y.to(self.device)
                    
                    # Forward pass
                    logits, loss = self.model(x, y)
                    total_loss += loss.item()
                    
                    # Update progress bar
                    pbar.set_postfix({'loss': loss.item()})
                    
        return total_loss / num_batches
        
    def train(
        self,
        num_epochs: int,
        save_every: int = 1,
        evaluate_every: int = 1
    ) -> None:
        """Train the model.
        
        Args:
            num_epochs (int): Number of epochs to train for
            save_every (int, optional): Save checkpoint every n epochs. Defaults to 1
            evaluate_every (int, optional): Evaluate every n epochs. Defaults to 1
        """
        try:
            for epoch in range(num_epochs):
                logger.info(f'Starting epoch {epoch + 1}/{num_epochs}')
                
                # Train
                train_loss = self.train_epoch()
                logger.info(f'Epoch {epoch + 1} - Training loss: {train_loss:.4f}')
                
                # Evaluate
                if self.val_loader and (epoch + 1) % evaluate_every == 0:
                    val_loss = self.evaluate()
                    logger.info(f'Epoch {epoch + 1} - Validation loss: {val_loss:.4f}')
                
                # Save checkpoint
                if (epoch + 1) % save_every == 0:
                    self.save_checkpoint(epoch + 1, train_loss)
                    
        except Exception as e:
            logger.error(f'Error during training: {str(e)}')
            raise 