Learning How LLMs Work

Part 1: Input data
Turn textx into tokens into token ids into token embeddings + positional embeddings = input embeddings

Tokens are turned into ids using a "known" vocabulary. In this case, using tiktoken. 

- Roughly how text preprocessing works (the manual regex way and using tiktoken)
- unknown token handling
- created a dataset used with DataLoader and created a dataloader

Part 2: Attention
Why self-attention? Other models needed to read full input and then "memorize" it. When translating text, the model can't look at full input and loses information. Self-attention can look at whole input by looking at the importance of different tokens. 

(The simple "not real" self-attention)
- Compute Attention Scores (Dot product between query and keys)
- Normalize using softmax function
- Context Vector: final result of self-attention for a token representing what information the token should carry

The real LLM attention mechanism

nn.Module is base Neural Network for Pytorch
super().__init__() for inheritance


Given embedding:
- every attention Layer has query, key, and value weights learned during training
- The embedding matrix is multipled by each weight matrix to get query (what a token looks for), key (what a token represents), value (what a token has)
- Attention score is the query and the key dot product (measuring similarity between a token and all other tokens), then normalized (scaled + softmax) to get weights. Weights then dot product with value to get final context vector. 

Causal Attention mask
- Used such that a token can only attend to iteslf and previous tokens, not future ones so during training, it can't "Cheat" 

Dropout Mask
- Masked out random positions to reduce overfitting
- Applied during training


Multi-Head Attention
- Each head has different weights, so they will learn different features
- Each head produces its own context vectors which are concatenated then applied a linear layer (out_proj) to mix them

Part 3: Architecture

- Layer Normalization: used to spped up training and stabilize by normalizing activations. For an input, layer computes mean and variance such that mean = 0 and variance = 1. 
    - Scale and Shift Weights in Layer Normalization are used to adjust the normalized output
- GELU: a non-linear activation function that sits in between linear layers. A FeedForward Module is coded into the transformer block that contains Linear Layer, GELU, and Linear Layer. 
- Activation functions add non-linearity to neural networks to learn complex patterns. 

nn.Sequential is a container that allows to build layers in neural nets
In a Linear layer, the input activations are used to learn weights using y = xW+b. The First Linear Layer expands the feature size, then GELU applies smooth filtering, then Second Layer compresses it and "summarizes" what is learned into the right dimensions. 

- Shortcut Connections: input is added directly to output to preserve original information and make gradients flow easier. 

Created the Transformer block and the GPT Model that includes the Transformer Block with everything combined. The Transformer is using MultiHeadAttention from model.py and builds TransformerBlock with LayerNorm, Attn, Dropout + shortcut. The Full Model stacks 12 Transformer vlocks, adding token and outputting logits

- RoPE Embeddings: Instead of absolute positional embeddings, using RoPE ebeddings precomputes rotation angles for each position up to max_seq_len. The model can understand when two tokens are "x positions" apart rather than token x attends to token at other random position

Part 4: Training

Loss
- Cross Entropy loss using torch.nn.functional.cross_entropy
- Perplexity (A more interpretable version that is e^loss)


Used very small training set(literally like one wiki article)
Seperate into training and validation set. Batch Sizing for training

Actual Training:
- In each epoch, iterate over batches
- Reset loss gradients from previous batch iterations, then calculate loss on current batch
- Backward pass to calculate loss gradients (using built in loss.backward())
- Before updating model weights, use gradient clipping
Gradient Clipping: ensures the magnitude stays within a certain threshold to avoid overstepping
- Update model weights (using built in optimizer.step()), Additionally, use a learning rate schedular (Usually starts with high learning rate, then slowly converges by lowering the learning rate)
- Check for training and validation set losses
- At the end, generate sample test
- Do for # of Epochs

Other Stuff for Training
- Sampling Procedures 
Temperature Scaling: Instead of using direct sampling from the distribution, you first scale them, making large values even larger. Therefore, model becomes more "confident" 
Sampling allows the outputs to be more varied. A argmax just always picks the highest which is boring. 
Top-k Sampling: Using torch.topk(), it is a way to control randomness by only picking the top-k most likely tokens before sampling. Thatway it still varies but still makes sense. 


To chat with LLM: python -m src.main --mode run