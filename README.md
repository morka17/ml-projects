# ML Projects

This repository contains various machine learning projects and implementations.

## Project Structure

```
.
├── makemore/              # Character-level language model for generating names
│   ├── data/             # Dataset files
│   ├── dataset.py        # Dataset handling utilities
│   ├── model.py          # Model architecture
│   └── train.py          # Training utilities
├── omr_sheet_grader/     # Optical Mark Recognition for grading sheets
│   ├── alphanumeric_detector.py  # Detector for alphanumeric characters
│   └── annotation.json   # Annotations for training data
├── gpt4-tokenizer/       # GPT-4 tokenizer implementation
├── tokenization/         # General tokenization utilities
├── nano-gpt/            # Minimal GPT implementation
└── datasets/            # Common datasets
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ml-projects.git
cd ml-projects
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Projects

### makemore

A character-level language model for generating names. The model uses a simple architecture with embedding layers and positional encoding to generate new names based on training data.

#### Usage Example

```python
from makemore import NameDataset, BigramLanguageModel, Trainer

# Load dataset
dataset = NameDataset('makemore/data/names.txt')

# Create model
model = BigramLanguageModel(
    vocab_size=dataset.vocab_size,
    n_embd=64,
    context_size=3
)

# Create trainer
trainer = Trainer(
    model=model,
    train_dataset=dataset,
    learning_rate=1e-3,
    batch_size=32
)

# Train model
trainer.train(num_epochs=10)
```

### OMR Sheet Grader

An Optical Mark Recognition system for grading answer sheets. The system uses computer vision and deep learning to detect and recognize alphanumeric characters and markings on answer sheets.

#### Usage Example

```python
from omr_sheet_grader import OMRDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = OMRDataset(image_paths, annotations)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

* Thanks to Andrej Karpathy for the inspiration and tutorials
* Thanks to the PyTorch team for their excellent framework
* Thanks to the open-source community for their contributions 