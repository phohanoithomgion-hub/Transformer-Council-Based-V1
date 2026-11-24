import os
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from datasets import load_dataset

from model import CouncilModel
from dataset import get_tokenizer
from config import get_config


class EvalDataset(Dataset):
    def __init__(self, samples, tokenizer, max_len=512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ex = self.samples[idx]
        prompt_text = ex["prompt"]
        answer_text = ex["answer"]

        prompt_ids_full = self.tokenizer(prompt_text, truncation=False, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids_full)

        enc = self.tokenizer(
            prompt_text + answer_text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        if prompt_len < self.max_len:
            labels[:prompt_len] = -100
        return {
            "input_ids": input_ids,
            "labels": labels,
            "prompt_len": torch.tensor(prompt_len, dtype=torch.long),
        }


def load_council_model(checkpoint_path, config, device, base_ckpt=None):
    full_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    phase_info = ""
    is_lora = False
    if isinstance(full_state, dict):
        phase_idx = full_state.get("phase_idx")
        step_in_phase = full_state.get("step_in_phase")
        curriculum_complete = full_state.get("curriculum_complete", False)
        is_lora = full_state.get("is_lora", False)
        if phase_idx is not None:
            phase_info = f" | phase={phase_idx + 1} | step_in_phase={step_in_phase}"
            if curriculum_complete:
                phase_info += " | curriculum=COMPLETE"
        if is_lora:
            phase_info += " | is_lora=True"

    if is_lora:
        if base_ckpt is None:
            base_ckpt = f"{config.checkpoint_dir}/pretrain_final.pt"
        if not os.path.exists(base_ckpt):
            raise FileNotFoundError(
                f"Checkpoint is LoRA but base not found at {base_ckpt}. "
                f"Specify --base_ckpt or run `python main.py merge` first."
            )
        print(f"[info] Loading LoRA checkpoint. Base: {base_ckpt}")
        base_state = torch.load(base_ckpt, map_location="cpu", weights_only=False)
        if isinstance(base_state, dict) and "model" in base_state:
            base_state = base_state["model"]
        base_state = {k.replace("_orig_mod.", ""): v for k, v in base_state.items()}

        model = CouncilModel(config)
        model.load_state_dict(base_state, strict=False)
        del base_state

        from lora import apply_lora, merge_lora
        apply_lora(model, rank=config.lora_rank, alpha=config.lora_alpha)
        lora_state = full_state["model"] if isinstance(full_state, dict) and "model" in full_state else full_state
        lora_state = {k.replace("_orig_mod.", ""): v for k, v in lora_state.items()}
        model.load_state_dict(lora_state, strict=False)
        merge_lora(model)
        model = model.to(device)
    else:
        state = full_state["model"] if isinstance(full_state, dict) and "model" in full_state else full_state
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        model = CouncilModel(config).to(device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[warn] missing keys: {len(missing)}")
        if unexpected:
            print(f"[warn] unexpected keys: {len(unexpected)}")

    print(f"[info] Loaded checkpoint: {os.path.basename(checkpoint_path)}{phase_info}")
    model.eval()
    return model


def build_valid_samples(tokenizer, num_samples=500):
    samples = []
    sources = [
        ("openai/gsm8k", "train", "question", "answer"),
        ("AI-MO/NuminaMath-CoT", "train", "problem", "solution"),
    ]
    per_source = max(1, num_samples // len(sources))
    for ds_name, split, q_key, a_key in sources:
        try:
            ds = load_dataset(ds_name, split=split, streaming=True)
        except Exception:
            continue
        count = 0
        for ex in ds:
            if count >= per_source:
                break
            q = ex.get(q_key)
            a = ex.get(a_key)
            if not q or not a:
                continue
            samples.append({
                "prompt": f"Problem: {q}\nAnswer:",
                "answer": a if ds_name == "openai/gsm8k" else f" {a}",
            })
            count += 1
    return samples


@torch.no_grad()
def evaluate_valid_loss(model, dataloader, device, num_recursion=2, force_full_recursion=True):
    total_loss = 0.0
    total_tokens = 0
    total_n_used = 0
    total_samples = 0
    desc = "valid_loss (adaptive)" if not force_full_recursion else f"valid_loss (n_rec={num_recursion})"

    for batch in tqdm(dataloader, desc=desc):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(
                input_ids, labels=labels,
                num_recursion=num_recursion,
                force_full_recursion=force_full_recursion,
            )
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_sum = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="sum"
        )
        n_tokens = (shift_labels != -100).sum().item()
        total_loss += loss_sum.item()
        total_tokens += n_tokens

        base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        n_used = getattr(base_model, "_last_n_rec_used", num_recursion)
        total_n_used += n_used * input_ids.size(0)
        total_samples += input_ids.size(0)

    avg_loss = total_loss / max(1, total_tokens)
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    avg_n_used = total_n_used / max(1, total_samples)
    return {
        "loss": avg_loss,
        "perplexity": ppl,
        "n_tokens": total_tokens,
        "avg_n_used": avg_n_used if not force_full_recursion else num_recursion,
        "force_full": force_full_recursion,
    }

@torch.no_grad()
def evaluate_multi_recursion(model, dataloader, device, rec_levels=(1, 2, 4)):
    """
    Test loss ở nhiều mức recursion khác nhau (force_full).
    Dùng để xem model đã adapt với từng depth chưa.
    """
    results = {}
    for n_rec in rec_levels:
        total_loss = 0.0
        total_tokens = 0
        for batch in tqdm(dataloader, desc=f"loss@n_rec={n_rec}"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(
                    input_ids, labels=labels,
                    num_recursion=n_rec,
                    force_full_recursion=True,
                )
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_sum = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="sum"
            )
            total_loss += loss_sum.item()
            total_tokens += (shift_labels != -100).sum().item()
        avg_loss = total_loss / max(1, total_tokens)
        results[f"n_rec_{n_rec}"] = {
            "loss": avg_loss,
            "perplexity": float(torch.exp(torch.tensor(avg_loss)).item()),
        }
        print(f"  n_rec={n_rec:2d} | loss={avg_loss:.4f} | ppl={results[f'n_rec_{n_rec}']['perplexity']:.2f}")
    return results


def extract_answer(text):
    for marker in ["Answer:", "answer:", "####", "\\boxed{"]:
        if marker in text:
            part = text.split(marker)[-1].strip()
            return part.split("\n")[0].strip()
    return text.strip().split("\n")[-1].strip()


def compute_reward(pred, gt):
    try:
        from math_verify import parse, verify
        gold = parse(gt)
        p = parse(pred)
        if gold and p and verify(gold, p):
            return 1.0
    except Exception:
        pass
    return 1.0 if pred.strip().lower() == gt.strip().lower() else 0.0


@torch.no_grad()
def evaluate_benchmark(model, tokenizer, device, benchmarks, max_new_tokens=256, temperature=0.0, best_n=None):
    results = {}
    for name, loader_fn in benchmarks.items():
        try:
            ds = loader_fn()
        except Exception as e:
            print(f"[skip] {name}: {e}")
            continue

        correct = 0
        total = 0
        for ex in tqdm(ds, desc=f"bench:{name}"):
            prompt = ex["prompt"]
            gt = ex["answer"]
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256, padding=False)
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)

            gen_fn = model.generate
            if hasattr(torch, "compiler") and hasattr(torch.compiler, "disable"):
                gen_fn = torch.compiler.disable(gen_fn)

            old_best = getattr(model.config, "best_N", 0)
            if best_n is not None:
                model.config.best_N = best_n

            out = gen_fn(
                input_ids,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 0.0,
                eos_token_id=tokenizer.eos_token_id,
            )
            model.config.best_N = old_best

            completion = tokenizer.decode(out[0][input_ids.size(1):], skip_special_tokens=True)
            pred = extract_answer(completion)
            correct += compute_reward(pred, gt)
            total += 1

        acc = correct / max(1, total)
        results[name] = {"accuracy": acc, "correct": correct, "total": total}
        print(f"[{name}] acc={acc:.4f} ({correct}/{total})")
    return results


def build_benchmarks():
    def gsm8k_loader():
        ds = load_dataset("openai/gsm8k", "main", split="test")
        return [
            {
                "prompt": f"Problem: {ex['question']}\nAnswer:",
                "answer": ex["answer"].split("####")[-1].strip(),
            }
            for ex in ds
        ]

    def math500_loader():
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        return [
            {
                "prompt": f"Problem: {ex['problem']}\nAnswer:",
                "answer": ex["answer"],
            }
            for ex in ds
        ]

    return {"gsm8k": gsm8k_loader, "math500": math500_loader}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["valid", "benchmark", "all", "multi_rec"])
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--base_ckpt", type=str, default=None,
                        help="Base checkpoint for LoRA (if --ckpt is LoRA)")
    parser.add_argument("--num_valid", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--best_n", type=int, default=None)
    parser.add_argument("--num_recursion", type=int, default=2)
    parser.add_argument("--force_full", action="store_true",
                        help="Force full recursion. Without this flag, model uses adaptive stopping.")
    parser.add_argument("--rec_levels", type=str, default="1,2,4,6,8,10,12",
                        help="Comma-separated rec levels for multi_rec mode")
    parser.add_argument("--out", type=str, default="eval_results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_config()
    tokenizer = get_tokenizer()
    config.vocab_size = len(tokenizer)

    model = load_council_model(args.ckpt, config, device, base_ckpt=args.base_ckpt)

    report = {}

    if args.mode in ("valid", "all"):
        samples = build_valid_samples(tokenizer, args.num_valid)
        ds = EvalDataset(samples, tokenizer, config.max_seq_len)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        report["valid"] = evaluate_valid_loss(
            model, dl, device,
            num_recursion=args.num_recursion,
            force_full_recursion=args.force_full,
        )
        info = f"[valid] loss={report['valid']['loss']:.4f} ppl={report['valid']['perplexity']:.2f}"
        if not args.force_full:
            info += f" | avg_n_used={report['valid']['avg_n_used']:.2f}"
        print(info)

    if args.mode == "multi_rec":
        samples = build_valid_samples(tokenizer, args.num_valid)
        ds = EvalDataset(samples, tokenizer, config.max_seq_len)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        rec_levels = [int(x.strip()) for x in args.rec_levels.split(",")]
        print(f"[info] Testing {len(rec_levels)} rec levels: {rec_levels}")
        report["multi_rec"] = evaluate_multi_recursion(model, dl, device, rec_levels=rec_levels)

    if args.mode in ("benchmark", "all"):
        bms = build_benchmarks()
        report["benchmark"] = evaluate_benchmark(
            model, tokenizer, device, bms,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            best_n=args.best_n,
        )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()