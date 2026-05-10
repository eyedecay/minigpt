import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader



class GPTDatasetV1(Dataset):
    """
    GPTDataset that creates training examples for LLM using sliding window

    Attributes:
        input_ids (list): List of tensor objects with input token sequences
        target_ids (list): input_ids shifted one position to the right
    """
    def __init__(self, txt, tokenizer, max_length, stride):
        """
        Initialization of dataset, tokenizing text and creating input/target pairs

        Args:
            txt (str): raw text
            tokenizer (tiktoken.Encoding): tokenizer
            max_length (int): max length of input/target
            stride (int): step size
        """
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i+1: i + 1 + max_length]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
        
    def __len__(self):
        """
        Returns length for testing purposes

        Returns:
            None
        """
        return len(self.input_ids)

    def __getitem__(self, idx):
        """
        Returns input and target tensors 

        Args:
            idx (int): index of sample

        Returns:
            (tuple): (input_ids, target_ids)
        """
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size = 4, max_length = 256, stride = 120, shuffle = True, drop_last = True, num_workers = 0):
    """
    Turns raw text into a PyTorch DataLoader that produces batches of input/target token sequences
    Args:
        txt (str)
        batch_size (int)
        max_length (int)
        stride (int)
        shuffle (bool)
        drop_last (bool)
        num_workers (int)
    
    Returns:
        (torch.utils.data.DataLoader) An object that takes a dataset and returns batches of tensors
    """

    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, drop_last = drop_last, num_workers = num_workers)

    return dataloader
