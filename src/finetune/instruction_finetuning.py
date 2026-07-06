import os
import torch
import tiktoken
from src.model import GPTModel
from src.train import train_model_simple
from src.finetune.dataset import create_dataloaders

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = tiktoken.get_encoding("gpt2")

train_loader, val_loader = create_dataloaders(tokenizer, batch_size=16)

model = GPTModel(GPT_CONFIG_124M).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
training_total_steps = len(train_loader)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_total_steps)


checkpoint = torch.load("test_model1.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

train_model_simple(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    num_epochs=1,
    eval_freq=500,
    eval_iter=1,
    start_context="Below is an instruction that describes a task.\n\n### Instruction:\nSay hello.\n\n### Response:\n",
    tokenizer=tokenizer,
)
