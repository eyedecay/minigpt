import numpy as np 

all_tokens = np.fromfile("pretrainingtokens.bin", dtype = np.uint16)
total_tokens = len(all_tokens)
print(total_tokens)

split = int(total_tokens * 0.90)

train_tokens = all_tokens[:split]
val_tokens = all_tokens[split:]

train_tokens.tofile("train_tokens.bin")
val_tokens.tofile("val_tokens.bin")