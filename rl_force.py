import os
import copy
import torch
import torch.nn.functional as F
import bitsandbytes as bnb
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

from model import CouncilModel
from dataset import StreamingRLDataset, get_tokenizer
from config import get_config
from log import TrainingLogger, sample_batch_output


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


def find_latest_rl_checkpoint(ckpt_dir):
    import glob
    ckpts = glob.glob(os.path.join(ckpt_dir, "rl_step_*.pt"))
    if not ckpts:
        return None, 0
    def get_step(path):
        try:
            return int(os.path.basename(path).split("_")[-1].split(".")[0])
        except:
            return 0
    ckpts.sort(key=get_step, reverse=True)
    return ckpts[0], get_step(ckpts[0])

def find_latest_sft_lora_checkpoint(ckpt_dir):
    import glob
    ckpts = glob.glob(os.path.join(ckpt_dir, "sft_lora_step_*.pt"))
    if not ckpts:
        return None, 0
    def get_step(path):
        try:
            return int(os.path.basename(path).split("_")[-1].split(".")[0])
        except Exception:
            return 0
    ckpts.sort(key=get_step, reverse=True)
    return ckpts[0], get_step(ckpts[0])


def find_latest_sft_checkpoint(ckpt_dir):
    import glob
    ckpts = glob.glob(os.path.join(ckpt_dir, "sft_step_*.pt"))
    if not ckpts:
        return None, 0
    def get_step(path):
        try:
            return int(os.path.basename(path).split("_")[-1].split(".")[0])
        except Exception:
            return 0
    ckpts.sort(key=get_step, reverse=True)
    return ckpts[0], get_step(ckpts[0])


def compute_format_reward(prediction, thinking_min_chars=20):
    """
    Strict format reward: 1.0 nếu output có đúng cấu trúc:
        Thinking:
        <reasoning>
        Answer:
        <final_answer>
    Không có Thinking, không có Answer, answer rỗng, thinking ngắn, hoặc
    answer không có ký tự alphanumeric -> 0.0.
    """
    if not prediction or not prediction.strip():
        return 0.0
    if "Thinking:" not in prediction:
        return 0.0
    if "Answer:" not in prediction:
        return 0.0
    thinking_pos = prediction.find("Thinking:")
    answer_pos = prediction.find("Answer:")
    if answer_pos < thinking_pos:
        return 0.0
    thinking_part = prediction[thinking_pos + len("Thinking:"):answer_pos].strip()
    answer_part = prediction[answer_pos + len("Answer:"):].strip()
    if not answer_part:
        return 0.0
    if not any(c.isalnum() for c in answer_part[:20]):
        return 0.0
    if len(thinking_part) < thinking_min_chars:
        return 0.0
    return 1.0


def compute_accuracy_reward(prediction, ground_truth):
    """
    Parse answer từ sau 'Answer:' rồi verify với math_verify.
    """
    if "Answer:" in prediction:
        pred = prediction.split("Answer:")[-1].strip()
    else:
        pred = prediction.strip()
    pred = pred.split("\n")[0].strip()
    if not pred:
        return 0.0
    try:
        from math_verify import parse, verify
        gold = parse(ground_truth)
        p = parse(pred)
        if gold and p and verify(gold, p):
            return 1.0
    except Exception:
        pass
    pg = pred.lower().replace(" ", "")
    gg = ground_truth.strip().lower().replace(" ", "")
    if not gg:
        return 0.0
    return 1.0 if (pg == gg or gg in pg) else 0.0


def compute_reward(prediction, ground_truth, thinking_min_chars=20):
    """
    Gate: format sai -> accuracy = 0. Format đúng -> mới tính accuracy.
    Trả về (format_reward, accuracy_reward).
    """
    fmt = compute_format_reward(prediction, thinking_min_chars)
    if fmt < 1.0:
        return fmt, 0.0
    acc = compute_accuracy_reward(prediction, ground_truth)
    return fmt, acc


def compute_self_logprobs(model, prompt_ids, gen_ids, force_n_rec):
    logits = model(gen_ids, num_recursion=force_n_rec, force_full_recursion=True)
    gen_start = prompt_ids.size(1)
    gen_logits = logits[:, gen_start-1:gen_start-1 + (gen_ids.size(1) - gen_start), :]
    logprobs = F.log_softmax(gen_logits, dim=-1)
    targets = gen_ids[:, gen_start:]
    return logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def compute_oracle_logprobs(model, prompt_ids, gt_texts, gen_ids, tokenizer, force_n_rec):
    B = prompt_ids.size(0)
    T_p = prompt_ids.size(1)
    T_gen = gen_ids.size(1) - T_p
    device = prompt_ids.device
    pad_id = tokenizer.pad_token_id

    gt_batch = []
    gt_lens = []
    for i in range(B):
        gt_enc = tokenizer(gt_texts[i], truncation=True, max_length=256, return_tensors="pt")
        gt_ids_i = gt_enc["input_ids"].squeeze(0).to(device)
        gt_batch.append(gt_ids_i)
        gt_lens.append(gt_ids_i.size(0))
    T_g_max = max(gt_lens)
    gt_padded = torch.full((B, T_g_max), pad_id, dtype=torch.long, device=device)
    for i, gt in enumerate(gt_batch):
        gt_padded[i, :gt_lens[i]] = gt

    full = torch.cat([prompt_ids, gt_padded, gen_ids[:, T_p:]], dim=1)
    logits = model(full, num_recursion=force_n_rec, force_full_recursion=True)

    results = []
    for i in range(B):
        T_g_i = gt_lens[i]
        gen_start = T_p + T_g_i
        gen_logits = logits[i:i+1, gen_start-1:gen_start-1+T_gen, :]
        logprobs = F.log_softmax(gen_logits, dim=-1)
        targets = gen_ids[i:i+1, T_p:]
        lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        results.append(lp)
    return torch.cat(results, dim=0)


def compute_oppo_advantages(self_logprobs, oracle_logprobs, rewards, beta0=0.5, scale=4.0):
    B, T = self_logprobs.shape
    device = self_logprobs.device

    log_ratio = oracle_logprobs - self_logprobs
    log_ratio = log_ratio.clamp(-10.0, 10.0)

    p = torch.full((B,), beta0, device=device)
    advantages = torch.zeros(B, T, device=device)

    for t in range(T):
        lr = log_ratio[:, t]
        logit_p = torch.log(p / (1 - p + 1e-8))
        p = torch.sigmoid(logit_p + lr)
        w = 4.0 * p * (1.0 - p)
        advantages[:, t] = w * lr

    reward_signal = 2.0 * rewards - 1.0
    advantages = advantages * reward_signal.unsqueeze(1) * scale
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages


def load_model_from_checkpoint(model, config, checkpoint_path, base_pretrain_override=None):
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    is_lora = False
    ckpt_base_path = None
    if isinstance(state, dict):
        is_lora = state.get("is_lora", False)
        ckpt_base_path = state.get("base_pretrain_path", None)
        if "model" in state:
            state = state["model"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

    if is_lora:
        import glob
        existing = glob.glob(os.path.join(config.checkpoint_dir, "pretrain_*"))
        print(f"[load] Available pretrain files in {config.checkpoint_dir}:")
        for f in existing:
            print(f"  - {os.path.basename(f)}")

        if base_pretrain_override is not None:
            base_path = base_pretrain_override
            print(f"[load] Using base_pretrain_override: {base_path}")
        elif ckpt_base_path is not None and os.path.exists(ckpt_base_path):
            base_path = ckpt_base_path
            print(f"[load] Using base path stored in checkpoint: {base_path}")
        else:
            base_path = f"{config.checkpoint_dir}/pretrain_final.pt"
            if not os.path.exists(base_path):
                base_path, _ = find_latest_pretrain_checkpoint(config.checkpoint_dir)

        if base_path is None or not os.path.exists(base_path):
            raise FileNotFoundError(
                f"LoRA checkpoint but no base pretrain found. "
                f"Available pretrain files: {[os.path.basename(f) for f in existing]}. "
                f"Use `--base_pretrain <path>` to specify manually."
            )
        print(f"[load] LoRA checkpoint, loading base from {base_path}")
        base_state = torch.load(base_path, map_location="cpu", weights_only=False)
        if isinstance(base_state, dict) and "model" in base_state:
            base_state = base_state["model"]
        base_state = {k.replace("_orig_mod.", ""): v for k, v in base_state.items()}
        model.load_state_dict(base_state, strict=False)
        del base_state

        from lora import apply_lora, merge_lora
        apply_lora(model, rank=config.lora_rank, alpha=config.lora_alpha)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[load] LoRA missing: {len(missing)}, unexpected: {len(unexpected)}")
        merge_lora(model)
        print(f"[load] LoRA merged into base")
    else:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[load] missing: {len(missing)}, unexpected: {len(unexpected)}")
    del state
    return model


def train_rl_force(args_list=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_pretrain", type=str, default=None,
                        help="Path to base pretrain checkpoint (required if LoRA base not found)")
    args = parser.parse_args(args_list or [])

    config = get_config()
    device = torch.device(config.device)
    torch.manual_seed(config.seed)

    force_n_rec = config.rl_force_recursion
    config.best_N = force_n_rec
    config.adaptive_inference = False

    tokenizer = get_tokenizer()
    config.vocab_size = len(tokenizer)
    config.eos_token_id = tokenizer.eos_token_id
    print(f"Tokenizer vocab_size = {config.vocab_size}")
    print(f"[rl_force] Forced recursion = {force_n_rec}")

    logger = TrainingLogger(config.log_dir, "rl_oppo_force")
    logger.info(f"Config: {config}")
    logger.info(f"Force recursion: {force_n_rec}")

    dataset = StreamingRLDataset(tokenizer, max_prompt_len=256, max_tokens=config.rl_max_tokens)
    dataloader = DataLoader(
        dataset,
        batch_size=config.rl_batch_size,
        num_workers=0,
        pin_memory=True,
    )

    model = CouncilModel(config).to(device)

    resume_ckpt, resume_step = find_latest_rl_checkpoint(config.checkpoint_dir)
    resume_state = None

    if resume_ckpt is not None:
        print(f"Found RL checkpoint: {resume_ckpt} (step {resume_step})")
        resume_state = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        ckpt_force_n = resume_state.get("rl_force_n_rec") if isinstance(resume_state, dict) else None
        if ckpt_force_n is not None and ckpt_force_n != force_n_rec:
            print(f"[warn] Checkpoint was trained with n_rec={ckpt_force_n}, now using n_rec={force_n_rec}")
        if "model" in resume_state:
            resume_state["model"] = {
                k.replace("_orig_mod.", ""): v for k, v in resume_state["model"].items()
            }
        model = load_model_from_checkpoint(model, config, resume_ckpt)
    else:
        sft_path = f"{config.checkpoint_dir}/sft_final.pt"
        sft_lora_path = f"{config.checkpoint_dir}/sft_lora_final.pt"
        pretrain_path = f"{config.checkpoint_dir}/pretrain_final.pt"

        sft_lora_step_path, sft_lora_step = find_latest_sft_lora_checkpoint(config.checkpoint_dir)
        sft_step_path, sft_step = find_latest_sft_checkpoint(config.checkpoint_dir)
        pretrain_step_path, pretrain_step = find_latest_pretrain_checkpoint(config.checkpoint_dir)

        if os.path.exists(sft_lora_path):
            print(f"Loading SFT LoRA final from {sft_lora_path}")
            model = load_model_from_checkpoint(model, config, sft_lora_path, base_pretrain_override=args.base_pretrain)
        elif sft_lora_step_path is not None:
            print(f"Loading SFT LoRA step checkpoint: {sft_lora_step_path} (step {sft_lora_step})")
            model = load_model_from_checkpoint(model, config, sft_lora_step_path, base_pretrain_override=args.base_pretrain)
        elif os.path.exists(sft_path):
            print(f"Loading SFT final from {sft_path}")
            model = load_model_from_checkpoint(model, config, sft_path)
        elif sft_step_path is not None:
            print(f"Loading SFT step checkpoint: {sft_step_path} (step {sft_step})")
            model = load_model_from_checkpoint(model, config, sft_step_path)
        elif os.path.exists(pretrain_path):
            print(f"Loading pretrain final from {pretrain_path}")
            model = load_model_from_checkpoint(model, config, pretrain_path)
        elif pretrain_step_path is not None:
            print(f"Loading latest pretrain checkpoint: {pretrain_step_path} (step {pretrain_step})")
            model = load_model_from_checkpoint(model, config, pretrain_step_path)
        else:
            print(f"WARNING: No checkpoint found. RL force will start from random init.")

    ref_model = copy.deepcopy(model).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    print("Created ref_model (frozen copy)")

    if os.environ.get("USE_COMPILE", "1") == "1":
        try:
            model = torch.compile(model, mode="default", fullgraph=False, dynamic=True)
            print("Compile OK (policy)")
        except Exception as e:
            print(f"Compile policy failed: {e}")
        try:
            ref_model = torch.compile(ref_model, mode="default", fullgraph=False, dynamic=True)
            print("Compile OK (ref)")
        except Exception as e:
            print(f"Compile ref failed: {e}")

    optimizer = bnb.optim.PagedAdamW8bit(model.parameters(), lr=config.rl_lr, betas=(0.9, 0.95))
    total_steps = config.rl_total_steps
    save_interval = max(1, int(total_steps * config.save_every_ratio))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: 0.5 * (1 + math.cos(math.pi * min(s / total_steps, 1.0)))
    )

    model.train()
    optimizer.zero_grad()
    step = 0

    if resume_state is not None:
        try:
            base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            base_model.load_state_dict(resume_state["model"], strict=False)
            optimizer.load_state_dict(resume_state["optimizer"])
            scheduler.load_state_dict(resume_state["scheduler"])
            step = resume_state["step"]
            print(f"Resumed RL force from step {step}")
        except Exception as e:
            print(f"Resume failed: {e}. Starting RL force from scratch (step 0).")
            step = 0

    for epoch in range(config.rl_epochs):
        pbar = tqdm(dataloader, desc=f"RL Force (n_rec={force_n_rec}) Epoch {epoch}")
        for batch in pbar:
            prompt_ids = batch["prompt_ids"].to(device)
            ground_truths = batch["ground_truth"]
            B = prompt_ids.size(0)

            model.eval()
            prompt_mask = batch["prompt_mask"].to(device)
            gen_ids = model.generate(
                prompt_ids,
                attention_mask=prompt_mask,
                max_new_tokens=config.rl_max_new_tokens,
                temperature=config.rl_temperature,
                eos_token_id=tokenizer.eos_token_id
            )
            gen_texts = tokenizer.batch_decode(gen_ids[:, prompt_ids.size(1):], skip_special_tokens=True)
            model.train()

            thinking_min = getattr(config, "rl_thinking_min_chars", 20)
            format_list = []
            accuracy_list = []
            for g, gt in zip(gen_texts, ground_truths):
                fmt, acc = compute_reward(g, gt, thinking_min)
                format_list.append(fmt)
                accuracy_list.append(acc)

            format_rewards = torch.tensor(format_list, device=device, dtype=torch.float32)
            accuracy_rewards = torch.tensor(accuracy_list, device=device, dtype=torch.float32)

            w_fmt = getattr(config, "rl_format_weight", 0.3)
            w_acc = getattr(config, "rl_accuracy_weight", 1.0)
            rewards = w_fmt * format_rewards + w_acc * accuracy_rewards

            with autocast("cuda", dtype=torch.bfloat16):
                self_logprobs = compute_self_logprobs(model, prompt_ids, gen_ids, force_n_rec)
                oracle_logprobs = compute_oracle_logprobs(model, prompt_ids, list(ground_truths), gen_ids, tokenizer, force_n_rec)
                with torch.no_grad():
                    ref_logprobs = compute_self_logprobs(ref_model, prompt_ids, gen_ids, force_n_rec)

            T = min(self_logprobs.size(1), oracle_logprobs.size(1), ref_logprobs.size(1))
            self_lp = self_logprobs[:, :T].float()
            oracle_lp = oracle_logprobs[:, :T].float()
            ref_lp = ref_logprobs[:, :T].float()

            gen_part = gen_ids[:, prompt_ids.size(1):][:, :T]
            eos_mask = (gen_part == tokenizer.eos_token_id)
            after_eos = torch.zeros_like(eos_mask)
            if T > 1:
                after_eos[:, 1:] = eos_mask[:, :-1].cumsum(dim=1) > 0
            valid_mask = (~after_eos).float()
            valid_count = valid_mask.sum().clamp(min=1.0)

            advantages = compute_oppo_advantages(
                self_lp.detach(), oracle_lp.detach(), rewards,
                config.oppo_beta0_prior, config.oppo_advantage_scale
            )
            advantages = advantages * valid_mask

            rewards_std = rewards.std().item()
            if rewards_std < 1e-4:
                loss_pg = (self_lp * 0.0).sum()
            else:
                ratio = torch.exp(self_lp - ref_lp)
                clipped = torch.clamp(ratio, 1 - config.rl_clip_epsilon, 1 + config.rl_clip_epsilon)
                pg = -torch.min(ratio * advantages, clipped * advantages)
                loss_pg = (pg * valid_mask).sum() / valid_count

            kl = ((ref_lp - self_lp) * valid_mask).sum() / valid_count
            kl = torch.clamp(kl, -1.0, 1.0)
            loss = (loss_pg + config.rl_beta * kl) / config.rl_grad_accum

            if loss.abs().item() > 10.0:
                print(f"[SAFETY] loss={loss.item():.2f} > 10, skipping step")
                optimizer.zero_grad(set_to_none=True)
                step += 1
                continue

            loss.backward()

            if (step + 1) % config.rl_grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            step += 1
            cur_loss = loss.item() * config.rl_grad_accum
            cur_kl = kl.item()
            cur_reward = rewards.mean().item()

            if step % 10 == 0:
                mean_fmt = format_rewards.mean().item()
                mean_acc = accuracy_rewards.mean().item()
                pbar.set_postfix({
                    "loss": f"{cur_loss:.4f}",
                    "kl": f"{cur_kl:.4f}",
                    "rew": f"{cur_reward:.2f}",
                    "fmt": f"{mean_fmt:.2f}",
                    "acc": f"{mean_acc:.2f}",
                    "rec": f"{force_n_rec}",
                })
                logger.log_step(step, {
                    "loss": cur_loss,
                    "kl": cur_kl,
                    "reward": cur_reward,
                    "format_reward": mean_fmt,
                    "accuracy_reward": mean_acc,
                    "rec": force_n_rec,
                    "mode": "rl_force",
                })
            else:
                pbar.set_postfix({
                    "loss": f"{cur_loss:.4f}",
                    "kl": f"{cur_kl:.4f}",
                    "rew": f"{cur_reward:.2f}",
                    "fmt": f"{format_rewards.mean().item():.2f}",
                    "acc": f"{accuracy_rewards.mean().item():.2f}",
                })

            log_every = getattr(config, "rl_log_every_n_samples", 10)
            if step % (config.sample_every_n_steps * log_every) == 0:
                try:
                    best_idx = int(torch.argmax(rewards).item())
                    prompt_i = prompt_ids[best_idx:best_idx+1]
                    gen_i = gen_ids[best_idx:best_idx+1]
                    prompt_text = tokenizer.decode(prompt_i[0], skip_special_tokens=True)
                    completion_text = tokenizer.decode(
                        gen_i[0][prompt_i.size(1):], skip_special_tokens=True
                    )
                    logger.log_sample(step, prompt_text, completion_text, ground_truths[best_idx])
                    logger.info(
                        f"[best sample] reward={rewards[best_idx].item():.2f} "
                        f"fmt={format_rewards[best_idx].item():.2f} "
                        f"acc={accuracy_rewards[best_idx].item():.2f}"
                    )
                except Exception as e:
                    logger.info(f"Sampling failed: {e}")

            if step % save_interval == 0 and step > 0:
                os.makedirs(config.checkpoint_dir, exist_ok=True)
                save_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                torch.save({
                    "model": save_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step,
                    "config": config,
                    "rl_force_n_rec": force_n_rec,
                }, f"{config.checkpoint_dir}/rl_step_{step}.pt")
                pct = step / total_steps * 100
                logger.info(f"Saved RL checkpoint at step {step} ({pct:.1f}%)")

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    final_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({
        "model": final_model.state_dict(),
        "config": config,
        "rl_force_n_rec": force_n_rec,
    }, f"{config.checkpoint_dir}/rl_final.pt")
    logger.info("RL (OPPO Force) completed.")
    return model


if __name__ == "__main__":
    train_rl_force()