import torch 
import tiktoken
from src.model import GPTModel 
from src.data import create_dataloader_v1 
from src.train import train_model_simple
import matplotlib.pyplot as plt

test_cfg = {
    "vocab_size": 50257,
    "context_length": 64,
    "emb_dim": 128,
    "n_heads": 4, 
    "n_layers": 2,
    "drop_rate": 0.1, 
    "qkv_bias": False, 
}

def test_train():
    text = "test" * 100
    train_loader = create_dataloader_v1(text, batch_size=2, max_length = 64, stride = 32, shuffle = True)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available else "cpu")      
    model = GPTModel(test_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-4, weight_decay = 0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    train_losses, val_losses, track_tokens_seen = train_model_simple(
            model = model,
            train_loader = train_loader,
            val_loader = train_loader,
            optimizer = optimizer,
            scheduler = scheduler,
            device = device,
            num_epochs = 10,
            eval_freq = 5,
            eval_iter = 1,
            start_context = "Hi",
            tokenizer = tiktoken.get_encoding("gpt2")
        )
    
    plt.figure(figsize = (10, 5))
    plt.plot(track_tokens_seen, train_losses, label = "training loss")
    plt.plot(track_tokens_seen, val_losses, label = "val loss")
    plt.show()

if __name__ == "__main__":
    test_train()