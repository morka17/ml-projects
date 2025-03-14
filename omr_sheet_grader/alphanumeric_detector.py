from torch.utils.data import  DataLoader, Dataset
from torchvision import transforms 
from PIL import Image 

import torch 



class OMRDataset(Dataset):
    def __init__(self, image_paths, annotations): 
        self.image_paths = image_paths
        self.annotations = annotations 
        self.transform = transforms.ToTensor()


    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)
        boxes = self.annotations[idx]["boxes"]

        # List of [x_min, y_min, x_max, y_max]
        labels = self.annotations[idx]["labels"]  # List of class IDs 
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)    
        }

        return image, target 


# Create dataset and dataloader 
dataset = OMRDataset(image_paths, annotations)
dataloaders = DataLoader(dataset, an)
