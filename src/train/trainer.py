import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from ..generation.generator import generate, text_to_token_ids, token_ids_to_text

def calc_loss_batch(input_batch, target_batch, model, device):
    """
    Calculates Cross Entropy Loss of a batch
    Args:
        input_batch: torch.Tensor
        target_batch: torch.Tensor
        model: torch.nn.Module
        device: torch.device
    Returns
    Loss:
        torch.Tensor
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits, _ = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches = None):
    """
    Computes average loss over DataLoader (mean loss of all batches) 

    Args:
        data_loader (DataLoader): Pytorch Dataloader (input_batch, target_batch) pairs
        model (nn.Module): nn model
        device (torch.device): device (cuda, mps, cpu)
        num_batches (int, optional): number of batches. Defaults to None.

    Returns:
        float: average loss
    """
    total_loss = 0. 
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))


    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    
    return total_loss / num_batches



def train_model_simple(model, train_loader, val_loader, optimizer, scheduler, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    """
    Trains the Model
    Args:
        model: (torch.nn.Module)
        train_loader: (torch.utilis.data.DataLoader)
        val_loader: (torch.utilis.data.DataLoader)
        optimizer: (torch Optimizer)
        scheduler: (torch Scheduler)
        device: (torch.device)
        num_epochs: (int)
        eval_freq: (int)
        eval_iter: (int)
        start_context: (str)
        tokenizer: tokenizer
    Returns:
        list, list, list (train_losses, val_losses, track_tokens_seen)

    """
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, 1


    #Main Training Pipeline
    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            with torch.amp.autocast(device_type = "cuda", dtype = torch.bfloat16):
                loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
            # Optimizer + Learning Rate Scheduler
            optimizer.step()
            scheduler.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch {epoch + 1} (Step {global_step:06d}): train loss {train_loss:.3f}, val loss {val_loss:.3f} ")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                },"finetuned-1.pth")

        generate_and_print_sample(model, tokenizer, device, start_context)
        
    
    # Checkpoint for saving weights
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        },
        "finetuned-1.pth"
    )
    return train_losses, val_losses, track_tokens_seen 



def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    testing for train and validation losses
    Args:
        model
        train_loader
        val_loader
        device
        eval_iter
    Returns:
        train_loss
        val_loss
    """
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches = eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches = eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.context_length
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(model = model, idx = encoded, max_new_tokens = 50, context_size = context_size)
    
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " " ))
    model.train()