import unittest
import torch
import os
from gpt4_tokenizer import GPT4Tokenizer

class TestGPT4Tokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = GPT4Tokenizer()
    
    def test_basic_encoding_decoding(self):
        text = "Hello, world!"
        tokens = self.tokenizer.encode(text)
        decoded_text = self.tokenizer.decode(tokens)
        self.assertEqual(text, decoded_text)
    
    def test_special_tokens(self):
        text = "<|im_start|>Hello<|im_end|>"
        tokens = self.tokenizer.encode(text)
        decoded_text = self.tokenizer.decode(tokens)
        self.assertEqual(text, decoded_text)
    
    def test_whitespace_handling(self):
        text = "Hello   world\n\t\r"
        tokens = self.tokenizer.encode(text)
        decoded_text = self.tokenizer.decode(tokens)
        self.assertEqual("Hello world", decoded_text)
    
    def test_unicode_characters(self):
        text = "Hello, 世界!"
        tokens = self.tokenizer.encode(text)
        decoded_text = self.tokenizer.decode(tokens)
        self.assertEqual(text, decoded_text)
    
    def test_token_counting(self):
        text = "Hello, world!"
        tokens = self.tokenizer.encode(text)
        count = self.tokenizer.count_tokens(text)
        self.assertEqual(len(tokens), count)
    
    def test_training(self):
        # Test data
        texts = [
            "Hello, world!",
            "This is a test sentence.",
            "Another example text.",
            "Testing the tokenizer training."
        ]
        
        # Train the tokenizer
        trainer = self.tokenizer.train(texts, vocab_size=1000, epochs=1, batch_size=2)
        
        # Test encoding after training
        text = "Hello, world!"
        tokens = self.tokenizer.encode(text)
        decoded_text = self.tokenizer.decode(tokens)
        self.assertEqual(text, decoded_text)
    
    def test_unknown_tokens(self):
        # Add some custom tokens to the vocabulary
        self.tokenizer.vocab[b'custom'] = 1000
        
        # Test text with unknown tokens
        text = "Hello custom world"
        tokens = self.tokenizer.encode(text)
        decoded_text = self.tokenizer.decode(tokens)
        
        # The unknown token should be handled gracefully
        self.assertIn(self.tokenizer.special_tokens['<|unknown|>'], tokens)
    
    def test_save_load(self):
        # Save the tokenizer
        save_path = "test_tokenizer.pt"
        self.tokenizer.save(save_path)
        
        # Create a new tokenizer and load the saved state
        new_tokenizer = GPT4Tokenizer()
        self.assertTrue(new_tokenizer.load(save_path))
        
        # Test encoding with the loaded tokenizer
        text = "Hello, world!"
        tokens = new_tokenizer.encode(text)
        decoded_text = new_tokenizer.decode(tokens)
        self.assertEqual(text, decoded_text)
        
        # Clean up
        if os.path.exists(save_path):
            os.remove(save_path)

if __name__ == '__main__':
    unittest.main() 