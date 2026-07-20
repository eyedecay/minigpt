# GPT-STYLE MiniGPT for Learning

<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/0dbf7c94-28ea-4f8d-8ff1-aaf844c7503d" />
<img width="400" height="400" alt="Screenshot 2026-07-20 at 5 30 38 PM" src="https://github.com/user-attachments/assets/3883e411-4785-49f7-9ea5-43b611ca8721" />


A GPT2 style language model 

Some explanations of what I implemented can be found [here](https://github.com/eyedecay/minigpt/blob/main/RANDOMEXPLANATIONS.md) (Not that its not fully exhaustive and I'll probably so back and fix some things but it should be enough to explain the training + implementation process) 

## Tech Stack
- Used Python + Pytorch for model training and finetuning, FastAPI for a simple connection to a React + Electron frontend.

## Installation
Download the latest zip in releases

### Local Installation
```bash
git clone https://github.com/eyedecay/minigpt.git
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

You also have to download the weights (finetuned-1.pth in releases) into the root of the repository

```bash
PYTHONPATH=. python backend/main.py
```
Open a new terminal
```bash
cd UI && npm install && npm run dev
```
Open a new terminal
```bash
cd UI && npm run electron

```

### AI Usage
- Was used to explain Pytorch-specific things and to build application at the end cause I ran into too many issues with pyinstaller and things like that.

[#horizons](horizons.hackclub.com)

- Model might take a bit to download weights/load backend server. If you wait like a minute then try prompting it it should work. 
- It's probably going to suck cause I didn't have enough credits to pretrain for too long (was renting GPU) but it should give answers that semantically are close to input.
