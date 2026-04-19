"""
Tokenization: splitting text into tokens (subwords/words) that a model can process.
"""

def simple_word_tokenizer(text):
    return text.lower().split()

def char_tokenizer(text):
    return list(text)

def bpe_like_tokenizer(text):
    """Simplified BPE: splits on punctuation and whitespace."""
    import re
    return re.findall(r"\w+|[^\w\s]", text.lower())

if __name__ == "__main__":
    text = "Hello, world! LLMs are amazing."

    print("Original text:", text)
    print()

    tokens = simple_word_tokenizer(text)
    print(f"Word tokens ({len(tokens)}):", tokens)

    tokens = char_tokenizer(text)
    print(f"Char tokens ({len(tokens)}):", tokens)

    tokens = bpe_like_tokenizer(text)
    print(f"BPE-like tokens ({len(tokens)}):", tokens)

    # Token IDs: assign each unique token an integer ID
    vocab = {tok: idx for idx, tok in enumerate(set(bpe_like_tokenizer(text)))}
    ids = [vocab[t] for t in bpe_like_tokenizer(text)]
    print()
    print("Vocabulary:", vocab)
    print("Token IDs :", ids)
