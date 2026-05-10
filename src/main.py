"""
Main Interface for development right now. 

To use, 
Run Model Locally
python -m src.main --mode run
Train Model
python -m src.main --mode train 
"""

import os
import torch
import tiktoken
import argparse
from src.model import GPTModel
from src.data import GPTDatasetV1, create_dataloader_v1
from src.train import train_model_simple
from src.generation import generate, text_to_token_ids, token_ids_to_text

GPT_CONFIG_124M = {
    "vocab_size": 50257, 
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12, 
    "n_layers": 12,
    "drop_rate": 0.1, 
    "qkv_bias": False, 
}

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available else "cpu")

def load_model(device):
    """
    Load the trained model from checkpoint for inference

    Args:
        device (torch.device): device used

    Returns:
        torch.nn.Module: GPTModel loaded with weights set to eval mode
    """

    model = GPTModel(GPT_CONFIG_124M).to(device)
    model.to(device)

    checkpoint = torch.load("model_and_optimizer.pth", map_location = device)
    model.load_state_dict(checkpoint["model_state_dict"])


    model.eval()
    return model


def run_model(device):
    """
    Runs model generation loop

    Args:
        device (torch.device): device
    """
    model = load_model(device)
    tokenizer = tiktoken.get_encoding("gpt2")

    prompt = input("\n Enter prompt: ")
    while prompt != "q":
        token_ids = text_to_token_ids(prompt, tokenizer).to(device)

        output = generate(
            model = model,
            idx = token_ids,
            max_new_tokens = 100,
            context_size = GPT_CONFIG_124M["context_length"]
        )
        print(token_ids_to_text(output, tokenizer))
        prompt = input("\n Enter prompt: ")

    print("Ended")
    

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available else "cpu")
    
    parser = argparse.ArgumentParser(description="Train or Run")
    parser.add_argument("--mode", choices = ["train", "run"], required = True, help = "Train or run")
    args = parser.parse_args()

    # Train mode 
    if args.mode == "train":
        #Placeholder text (find real data)
        with open("the-verdict.txt", "r", encoding = "utf-8") as f:
            raw_text = f.read()
        
        train_loader = create_dataloader_v1(raw_text, batch_size=2, max_length = GPT_CONFIG_124M["context_length"], stride = GPT_CONFIG_124M["context_length"], shuffle = True)
        val_loader = create_dataloader_v1(raw_text, batch_size=2, max_length = GPT_CONFIG_124M["context_length"], stride = GPT_CONFIG_124M["context_length"], shuffle = False)
        
        model = GPTModel(GPT_CONFIG_124M).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-4, weight_decay = 0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        #start using saved weights 
        if os.path.exists("model_and_optimizer.pth"):
            print("resume training")
            checkpoint = torch.load("model_and_optimizer.pth", map_location = device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            print("Training from start")

        
        train_model_simple(
            model = model,
            train_loader = train_loader,
            val_loader = val_loader,
            optimizer = optimizer,
            scheduler = scheduler,
            device = device,
            num_epochs = 10,
            eval_freq = 5,
            eval_iter = 1,
            start_context = "Hi",
            tokenizer = tiktoken.get_encoding("gpt2")
        )


    #test model locally
    elif args.mode == "run":
        run_model(device)


if __name__ == "__main__":
    main()
    

