# INJA: INdirect LLM JAilbreak by leveraging the visual modality

This is the offical implementation of paper: **"Efficient Indirect LLM Jailbreak by Leveraging Visual Modality"**

## Repository Structure

```text
.
├── jb.py                         # Main experiment entry script
├── requirements.txt              # Python dependencies used by this repo
├── dataset/
│   ├── advbench/
│   ├── advimage/
│   ├── harmbench/
│   └── jailbreakbench/
├── results/                      # Output logs and results
├── torchattacks/                 # Attack and optimization utilities
└── minigpt4/                     # Expected MiniGPT-4 package path
```

## Getting Started

#### Clone the Repository

```bash
git clone https://github.com/LZzz2000/INJA.git
cd INJA
```

#### Create the Python Environment

```bash
conda create -n INJA python=3.10 -y
conda activate INJA
pip install --upgrade pip
pip install -r requirements.txt
```

#### Install MiniGPT-4

This project expects the MiniGPT-4 codebase and configuration layout.

```bash
git clone https://github.com/Vision-CAIR/MiniGPT-4.git
```

Follow the official MiniGPT-4 README to download:

- the base LLM weights required by the chosen MiniGPT-4 variant
- the corresponding pretrained MiniGPT-4 checkpoint

#### Prepare the Guard Model

This repository leverages **Meta-Llama-Guard-2-8B** to automatically evaluate response safety. Access to the model requires approval via Hugging Face.

#### Run

```python
python jb.py --start_idx 0 --end_idx 10 --save_dir ./results/jailbreakbench/LLaMA
```

#### Outputs

Typical outputs include:

- log files
- evaluation CSV files
- jailbreak JSON files
