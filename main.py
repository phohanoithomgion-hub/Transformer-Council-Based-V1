import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,roundup_power2_divisions:4"
import sys
from train_sft import train_sft
from rl_force import train_rl_force
from pretraining import train_pretrain
from inference import run_inference
from merge_lora import run_merge

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [pretrain|sft|merge|rl|inference] [args...]")
        return
    mode = sys.argv[1].lower()
    if mode == "pretrain":
        train_pretrain(sys.argv[2:])
    elif mode == "sft":
        train_sft(sys.argv[2:])
    elif mode == "merge":
        run_merge(sys.argv[2:])
    elif mode == "rl":
        train_rl_force(sys.argv[2:])
    elif mode == "inference":
        run_inference(sys.argv[2:])
    else:
        print(f"Unknown mode: {mode}")

if __name__ == "__main__":
    main()