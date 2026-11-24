import os
import json
import logging
from datetime import datetime


class TrainingLogger:
    def __init__(self, log_dir="logs", name="train"):
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"{name}_{ts}.log")
        self.jsonl_file = os.path.join(log_dir, f"{name}_{ts}.jsonl")
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        fmt = logging.Formatter("%(asctime)s | %(message)s")
        fh = logging.FileHandler(self.log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        self.logger.addHandler(fh)
        self.logger.addHandler(sh)
        self.info(f"Logger initialized. Log: {self.log_file}")
        self.info(f"JSONL: {self.jsonl_file}")

    def info(self, msg):
        self.logger.info(msg)

    def log_step(self, step, data):
        parts = []
        for k, v in data.items():
            if isinstance(v, float):
                if v != 0.0 and abs(v) < 1e-3:
                    parts.append(f"{k}={v:.3e}")
                else:
                    parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        self.info(f"STEP {step} | " + " | ".join(parts))
        with open(self.jsonl_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step, **data}, ensure_ascii=False) + "\n")

    def log_probe(self, step, result):
        losses = result['loss_per_rec'].tolist()
        optimal_N = result['optimal_N']
        per_sample_N = result['per_sample_optimal_N'].tolist()
        lines = [f"STEP {step} | PROBE | optimal_N={optimal_N} | per_sample_N={per_sample_N}"]
        for k, l in enumerate(losses, 1):
            lines.append(f"  loss@{k}={l:.4f}")
        self.info("\n".join(lines))
        with open(self.jsonl_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "step": step,
                "type": "probe",
                "loss_per_rec": losses,
                "optimal_N": optimal_N,
                "per_sample_N": per_sample_N,
            }, ensure_ascii=False) + "\n")

    def log_sample(self, step, prompt, completion, ref=None):
        self.info(f"\n{'='*60}\n[STEP {step}] SAMPLE OUTPUT\n{'-'*60}")
        self.info(f"PROMPT:\n{prompt}")
        self.info(f"COMPLETION:\n{completion}")
        if ref is not None:
            self.info(f"GROUND-TRUTH:\n{ref}")
        self.info(f"{'='*60}\n")
        with open(self.jsonl_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "step": step,
                "type": "sample",
                "prompt": prompt,
                "completion": completion,
                "reference": ref
            }, ensure_ascii=False) + "\n")


def sample_batch_output(model, tokenizer, batch, device, max_new_tokens=64, temperature=0.1):
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    was_training = base_model.training
    base_model.eval()
    try:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        if "prompt_ids" in batch:
            prompt_ids = batch["prompt_ids"][:1].to(device)
            prompt_mask = batch["prompt_mask"][:1].to(device)
        else:
            content_len = (labels[0] != -100).sum().item()
            prompt_len = max(50, content_len // 3)
            prompt_ids = input_ids[:1, :prompt_len]
            prompt_mask = (prompt_ids != tokenizer.pad_token_id).long()

        prompt_text = tokenizer.decode(prompt_ids[0], skip_special_tokens=True)

        import torch as _torch
        gen_fn = base_model.generate
        if hasattr(_torch, "compiler") and hasattr(_torch.compiler, "disable"):
            gen_fn = _torch.compiler.disable(gen_fn)

        with _torch.no_grad():
            output = gen_fn(
                prompt_ids,
                attention_mask=prompt_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                eos_token_id=tokenizer.eos_token_id
            )
        completion_text = tokenizer.decode(
            output[0][prompt_ids.size(1):], skip_special_tokens=True
        )
        return prompt_text, completion_text
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"[sampling failed: {e}]"
    finally:
        if was_training:
            base_model.train()