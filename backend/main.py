from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import torch

from src.model import GPTModel
from src.generation import (generate, text_to_token_ids, token_ids_to_text)
import tiktoken


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
tokenizer = tiktoken.get_encoding("gpt2")

def load_model(device):
    """
    Load the trained model from checkpoint for inference

    Args:
        device (torch.device): device used

    Returns:
        torch.nn.Module: GPTModel loaded with weights set to eval mode
    """

    model = GPTModel(GPT_CONFIG_124M).to(device)


    checkpoint = torch.load("finetuned-1.pth", map_location = device)
    model.load_state_dict(checkpoint["model_state_dict"])


    model.eval()
    #model = torch.compile(model)
    return model



model = load_model(device)

app = FastAPI()

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Prompt(BaseModel):
    prompt: str

@app.get("/")
def root():
    return {"Hello": "World"}

@app.post("/generate")
def generate_response(request: Prompt):


    prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the Request. \n ### Instruction: \n {request.prompt} \n ###Response: "
    token_ids = text_to_token_ids(prompt, tokenizer).to(device)

    output = generate(
        model = model,
        idx = token_ids,
        max_new_tokens = 100,
        context_size = GPT_CONFIG_124M["context_length"],
        temperature = 0.8,
        top_k = 50,
        eos_id = 50256,
        repetition = 1.3
    )
    response = token_ids_to_text(output[:, token_ids.shape[-1]:], tokenizer)

    return {
        "response": response
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
        


        

