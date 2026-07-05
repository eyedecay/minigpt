import torch 
import tiktoken 
from datasets import load_dataset 
from torch.utils.data import Dataset 

tokenizer = tiktoken.get_encoding("gpt2")

class AlpacaDataset(Dataset):
    def __init__(self, split_data,):
        self.examples = []
        for example in split_data:
            prompt = format_prompt(example["instruction"], example["input"])

            response = example["output"]

            prompt_ids = tokenizer.encode(prompt)
            response_ids = tokenizer.encode(response)

            input_ids = prompt_ids + response_ids + [50256]

            target_ids = ([-100] * len(prompt_ids) + response_ids + [50256])

            self.examples.append((torch.tensor(input_ids, dtype = torch.long), torch.tensor(target_ids, dtype = torch.long)))
            
def format_prompt(instruction, input_text):
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



