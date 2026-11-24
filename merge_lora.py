import os
import argparse
import torch
from config import get_config
from dataset import get_tokenizer
from model import CouncilModel
from lora import apply_lora, merge_lora

def find_latest_pretrain_checkpoint(ckpt_dir):
    import glob
    ckpts = glob.glob(os.path.join(ckpt_dir, "pretrain_curriculum_*.pt"))
    if not ckpts:
        ckpts = glob.glob(os.path.join(ckpt_dir, "pretrain_step_*.pt"))
    if not ckpts:
        return None, 0
    def get_step(path):
        try:
            return int(os.path.basename(path).split("_")[-1].split(".")[0])
        except Exception:
            return 0
    ckpts.sort(key=get_step, reverse=True)
    return ckpts[0], get_step(ckpts[0])

def run_merge(args_list):
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_ckpt", type=str, required=True)
    parser.add_argument("--pretrain_ckpt", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args(args_list)

    config = get_config()
    tokenizer = get_tokenizer()
    config.vocab_size = len(tokenizer)

    if args.pretrain_ckpt is not None:
        pretrain_path = args.pretrain_ckpt
    else:
        final_path = f"{config.checkpoint_dir}/pretrain_final.pt"
        if os.path.exists(final_path):
            pretrain_path = final_path
        else:
            pretrain_path, _ = find_latest_pretrain_checkpoint(config.checkpoint_dir)
    if pretrain_path is None or not os.path.exists(pretrain_path):
        raise FileNotFoundError("Cannot find pretrain checkpoint to use as base")

    print(f"[merge] Loading base from {pretrain_path}")
    base = CouncilModel(config)
    base_state = torch.load(pretrain_path, map_location="cpu", weights_only=False)
    if isinstance(base_state, dict) and "model" in base_state:
        base_state = base_state["model"]
    base_state = {k.replace("_orig_mod.", ""): v for k, v in base_state.items()}
    base.load_state_dict(base_state, strict=False)
    del base_state

    print(f"[merge] Applying LoRA rank={config.lora_rank}, alpha={config.lora_alpha}")
    apply_lora(base, rank=config.lora_rank, alpha=config.lora_alpha)

    print(f"[merge] Loading LoRA weights from {args.lora_ckpt}")
    lora_state = torch.load(args.lora_ckpt, map_location="cpu", weights_only=False)
    best_N = lora_state.get("best_N", 0) if isinstance(lora_state, dict) else 0
    if isinstance(lora_state, dict) and "model" in lora_state:
        lora_state = lora_state["model"]
    lora_state = {k.replace("_orig_mod.", ""): v for k, v in lora_state.items()}
    base.load_state_dict(lora_state, strict=False)
    del lora_state

    print("[merge] Merging LoRA into base weights...")
    merge_lora(base)

    out_path = args.output or f"{config.checkpoint_dir}/sft_final.pt"
    torch.save({
        "model": base.state_dict(),
        "best_N": best_N,
        "config": config,
    }, out_path)
    print(f"[merge] Merged model saved to {out_path}")
    return out_path

if __name__ == "__main__":
    import sys
    run_merge(sys.argv[1:])