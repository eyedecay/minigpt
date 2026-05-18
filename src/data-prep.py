"""
Short script to take allenai dataset and get training data
"""

import numpy as np
from datasets import load_dataset 
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

eot_token = tokenizer.encode("<|endoftext|>", allowed_special = {"<|endoftext|>"})[0] 

dataset = load_dataset("allenai/c4", "realnewslike", streaming = True, split = "train")
total_tokens = 0

with open("pretrainingtokens.bin", "wb") as f:

    for i, sample in enumerate(dataset):
        try: 
            tokens = tokenizer.encode(sample["text"], allowed_special = {"<|endoftext|>"})
            tokens += [eot_token]
            np.array(tokens, dtype = np.uint16).tofile(f)
            total_tokens += len(tokens)

            if (i + 1) % 10000 == 0:
                print(f"processed {(i+1)} samples, {total_tokens} tokens")
        except Exception as e:
            print(f"skipping")
            continue




