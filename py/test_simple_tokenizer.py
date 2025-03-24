import simple_tokenizer

tokenizer = simple_tokenizer.Tokenizer("tokenizer.json")
print(tokenizer.tokenize("true true false"))
print(tokenizer.tokenize_to_strings("true true false"))
