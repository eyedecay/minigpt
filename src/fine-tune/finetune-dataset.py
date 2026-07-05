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


dataset = load_dataset("tatsu-lab/alpaca")
train_data = dataset["train"]
print(train_data[0])

def tokenize(example):
    tokens = tokenizer.encode(example["text"])
    return {"input_ids": tokens}

tokenized_dataset = train_data.map(tokenize, remove_columns=train_data.column_names)