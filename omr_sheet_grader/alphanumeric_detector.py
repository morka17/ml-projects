from torch.utils.data import DataLoader, Dataset
from torchvision import transforms 
from PIL import Image 
import torch 
import os
from typing import List, Dict, Any, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OMRDataset(Dataset):
    """Dataset class for Optical Mark Recognition (OMR) sheet processing.
    
    This class handles loading and preprocessing of images and their corresponding
    annotations for training an object detection model to recognize alphanumeric
    characters on OMR sheets.
    
    Args:
        image_paths (List[str]): List of paths to the input images
        annotations (List[Dict]): List of annotation dictionaries containing bounding boxes and labels
        transform (transforms.Compose, optional): Custom transform pipeline. If None, uses ToTensor
    """
    
    def __init__(self, 
                 image_paths: List[str], 
                 annotations: List[Dict[str, Any]], 
                 transform: transforms.Compose = None) -> None:
        if len(image_paths) != len(annotations):
            raise ValueError("Number of images and annotations must match")
            
        self.image_paths = image_paths
        self.annotations = annotations
        self.transform = transform if transform is not None else transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Validate index
        if idx < 0 or idx >= len(self.image_paths):
            raise IndexError(f"Index {idx} out of range")
            
        try:
            # Load and convert image
            image_path = self.image_paths[idx]
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found at {image_path}")
                
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)
            
            # Get annotations
            boxes = self.annotations[idx]["boxes"]
            labels = self.annotations[idx]["labels"]
            
            # Create target dictionary
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64)    
            }
            
            return image, target
            
        except Exception as e:
            logger.error(f"Error processing item {idx}: {str(e)}")
            raise

def create_dataloader(
    dataset: OMRDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """Creates a DataLoader for the OMR dataset.
    
    Args:
        dataset (OMRDataset): The dataset to create a loader for
        batch_size (int, optional): Number of samples per batch. Defaults to 32
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True
        num_workers (int, optional): Number of worker processes. Defaults to 4
        pin_memory (bool, optional): Whether to pin memory in GPU training. Defaults to True
        
    Returns:
        DataLoader: The configured data loader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

# Create dataset and dataloader 
dataset = OMRDataset(image_paths, annotations)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
