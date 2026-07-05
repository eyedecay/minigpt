""" Instruction Fine-tuning on Alpaca Dataset"""

from datasets import load_dataset 

dataset = load_dataset("tatsu-lab/alpaca", split = "train")

