import torch 
import tiktoken 
from datasets import load_dataset 
from torch.utils.data import Dataset, DataLoader


class AlpacaDataset(Dataset):
    def __init__(self, split_data, tokenizer):
        self.examples = []
        self.tokenizer = tokenizer
        for example in split_data:
            prompt = format_prompt(example["instruction"], example["input"])

            response = example["output"]

            prompt_ids = tokenizer.encode(prompt)
            response_ids = tokenizer.encode(response)

            input_ids = prompt_ids + response_ids + [50256]

            target_ids = ([-100] * len(prompt_ids) + response_ids + [50256])

            self.examples.append((torch.tensor(input_ids, dtype = torch.long), torch.tensor(target_ids, dtype = torch.long)))
    
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        return self.examples[idx]

def format_prompt(instruction, input_text):
    """
    Format the prompt into one string

    Args:
        instruction (str): Instruction column for example in dataset
        input_text (str): input column for example in dataset (could not exist)

    Returns:
        str: formatted prompt
    """
    if input_text.strip():
        return (
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            f"### Instruction: \n {instruction} \n\n"
            f"### Input: \n {input_text} \n\n"
            "### Response:"
        )
    else:
        return (
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            f"### Instruction: \n {instruction} \n\n"
            "### Response:"
        )


def collate_fn(batch):
    """
    Combine individual samples into a batch
    Args:
        batch (List[Tuple[Tensor, Tensor]])
    """
    input_ids, target_ids = zip(*batch)
    max_len = max(len(x) for x in input_ids)

    padded_inputs = []
    padded_outputs = []

    for input, target in zip(input_ids, target_ids):
        pad = max_len - len(input)

        padded_inputs.append(torch.cat([input, torch.full((pad,), 50256, dtype = torch.long)]))
        padded_outputs.append(torch.cat([target, torch.full((pad,), -100, dtype = torch.long)]))
    
    return (torch.stack(padded_inputs), torch.stack(padded_outputs))

def create_dataloaders(tokenizer, batch_size = 16, train_ratio = 0.9, seed = 42):
    """
    Create the finetuning dataloaders for train and val

    Args:
        tokenizer (tiktoken.encoding): gpt2 encoding
        batch_size (int, optional): batch size. Defaults to 16.
        train_ratio (float, optional): 90 10 split of train and val. Defaults to 0.9.
        seed (int, optional): random seed. Defaults to 42.

    Returns:
        train_loader, val_loader: dataloaders 
    """
    dataset = load_dataset("tatsu-lab/alpaca")["train"]
    dataset = dataset.shuffle(seed = seed)

    split = int(len(dataset) * train_ratio)

    train_dataset = AlpacaDataset(dataset.select(range(split)), tokenizer)
    val_dataset = AlpacaDataset(dataset.select(range(split, len(dataset))), tokenizer)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, collate_fn = collate_fn)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True, collate_fn = collate_fn)

    return train_loader, val_loader