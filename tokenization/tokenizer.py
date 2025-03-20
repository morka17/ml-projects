import re 






if __name__ == "__main__":


    training_text = """
    The quick brown fox jumps over the lazy dog/
    The quick brown fox jumps again and again .

    """


    # Initilize and train tokenizer 
    tokenizer = SimpleBPETokenizer()
    tokenizer.train(training_text, num_merges=10)


    # Test tokenization 
    test_text = "The quick for jumps"
    tokens = tokenizer.tokenize(test_text)
    print("Tokens:", tokens)


    # Decode tokens 
    revers_vocab = {v: k for k, v in tokenizer.vocab.items()}
    deoced = [reverse_vocab.get(token, '<unk>') for token in tokens]
    print("Decoed tokens:", decoded)
