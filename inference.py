import os
import argparse
import torch

from model import CouncilModel
from dataset import get_tokenizer
from config import get_config


def load_council_model(checkpoint_path, config, device):
    model = CouncilModel(config).to(device)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    best_N = 0
    if isinstance(state, dict) and "model" in state:
        best_N = state.get("best_N", 0)
        state = state["model"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)}")
    model.eval()
    return model, best_N


@torch.no_grad()
def generate_text(model, tokenizer, prompt, device, max_new_tokens=256, temperature=0.7, repetition_penalty=1.15, no_repeat_ngram_size=3):
    ids = tokenizer.encode(prompt, truncation=True, max_length=512)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attn = torch.ones_like(input_ids)

    if input_ids.size(1) < 3:
        print(f"[warn] prompt is only {input_ids.size(1)} token(s), output quality will be poor")

    gen_fn = model.generate
    if hasattr(torch, "compiler") and hasattr(torch.compiler, "disable"):
        gen_fn = torch.compiler.disable(gen_fn)

    out = gen_fn(
        input_ids,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        temperature=temperature if temperature > 0 else 1e-3,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )
    completion = tokenizer.decode(out[0][input_ids.size(1):], skip_special_tokens=True)
    return completion


def run_inference(args_list):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num_recursion", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=None)
    args = parser.parse_args(args_list)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = get_config()
    tokenizer = get_tokenizer()
    config.vocab_size = len(tokenizer)

    rep_pen = args.repetition_penalty if args.repetition_penalty is not None else config.inference_repetition_penalty
    nrng = args.no_repeat_ngram_size if args.no_repeat_ngram_size is not None else config.inference_no_repeat_ngram_size

    model, ckpt_best_N = load_council_model(args.ckpt, config, device)

    if args.num_recursion is not None:
        config.best_N = args.num_recursion
        print(f"[info] num_recursion overridden to {args.num_recursion}")
    elif ckpt_best_N > 0:
        config.best_N = ckpt_best_N
        print(f"[info] loaded best_N={ckpt_best_N} from checkpoint")
    elif "pretrain" in os.path.basename(args.ckpt).lower():
        config.best_N = 1
        print("[info] detected pretrain checkpoint, defaulting num_recursion=1")
    else:
        print(f"[warn] no best_N in checkpoint, using default {config.rl_generate_recursion}")

    print(f"[info] device={device}, ckpt={args.ckpt}")
    print(f"[info] repetition_penalty={rep_pen}, no_repeat_ngram_size={nrng}")
    print("=" * 60)

    if args.prompt is not None:
        completion = generate_text(
            model, tokenizer, args.prompt, device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            repetition_penalty=rep_pen,
            no_repeat_ngram_size=nrng,
        )
        print(f"PROMPT:\n{args.prompt}")
        print("-" * 60)
        print(f"COMPLETION:\n{completion}")
        print("=" * 60)
        return

    print("Interactive mode. Type 'quit' or 'exit' to stop.")
    print("=" * 60)
    while True:
        try:
            prompt = input("\n>>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.strip().lower() in ("quit", "exit"):
            break
        if not prompt.strip():
            continue
        completion = generate_text(
            model, tokenizer, prompt, device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            repetition_penalty=rep_pen,
            no_repeat_ngram_size=nrng,
        )
        print("-" * 60)
        print(completion)
        print("-" * 60)