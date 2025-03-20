# GPT-4 Tokenizer

A Python implementation of the GPT-4 tokenizer from scratch. This implementation follows the same tokenization approach used by GPT-4, which is based on the BPE (Byte-Pair Encoding) algorithm with some modifications.

## Features

- Implements GPT-4 tokenization algorithm
- Supports both encoding (text to tokens) and decoding (tokens to text)
- Handles special tokens and whitespace
- Provides token counting functionality

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from gpt4_tokenizer import GPT4Tokenizer

# Initialize the tokenizer
tokenizer = GPT4Tokenizer()

# Encode text to tokens
text = "Hello, world!"
tokens = tokenizer.encode(text)

# Decode tokens back to text
decoded_text = tokenizer.decode(tokens)

# Count tokens in text
token_count = tokenizer.count_tokens(text)
```

## Implementation Details

This implementation:
1. Uses regex for text preprocessing
2. Implements BPE (Byte-Pair Encoding) algorithm
3. Handles special tokens and whitespace according to GPT-4 specifications
4. Provides efficient token counting

## License

MIT License 