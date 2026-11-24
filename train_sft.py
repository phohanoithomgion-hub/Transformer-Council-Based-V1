import os
import math
import torch
import torch.nn.functional as F
import bitsandbytes as bnb
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import CouncilModel
from dataset import StreamingSFTDataset, get_tokenizer
from config import get_config
from log import TrainingLogger, sample_batch_output
from lora import apply_lora, freeze_and_unfreeze_lora, count_trainable

torch._dynamo.config.cache_size_limit = 64
torch.backends.cudnn.benchmark = True

def find_latest_checkpoint(ckpt_dir, use_lora=False):
    import glob
    if use_lora:
        ckpts = glob.glob(os.path.join(ckpt_dir, "sft_lora_step_*.pt"))
    else:
        ckpts = glob.glob(os.path.join(ckpt_dir, "sft_step_*.pt"))
    if not ckpts:
        return None, 0
    # Parse step từ tên file
    def get_step(path):
        try:
            return int(os.path.basename(path).split("_")[-1].split(".")[0])
        except:
            return 0
    ckpts.sort(key=get_step, reverse=True)
    return ckpts[0], get_step(ckpts[0])


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

def warmup_cosine(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))

def get_recursion_for_step(step, total_steps, config):
    if step < 2000:
        return 1
    if step < 4000:
        return 2
    if step < 6000:
        return 3
    if step < 8000:
        return 4
    if step < 10000:
        return 6
    if step < 12000:
        return 8
    if step < 14000:
        return 10
    return config.max_recursion


def train_sft(args_list=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_ckpt", type=str, default=None)
    parser.add_argument("--no_pretrain", action="store_true")
    args = parser.parse_args(args_list or [])

    config = get_config()
    device = torch.device(config.device)
    torch.manual_seed(config.seed)

    tokenizer = get_tokenizer()
    # SỬA LỖI 1: Ghi đè vocab_size từ tokenizer thực tế
    config.vocab_size = len(tokenizer)
    config.eos_token_id = tokenizer.eos_token_id
    print(f"Tokenizer vocab_size = {config.vocab_size}")

    logger = TrainingLogger(config.log_dir, "sft")
    logger.info(f"Config: {config}")

    dataset = StreamingSFTDataset(tokenizer, config.max_seq_len, config.sft_max_tokens)
    dataloader = DataLoader(
        dataset,
        batch_size=config.sft_batch_size,
        num_workers=0,
        pin_memory=True,
    )

    config.backbone_checkpoint = False
    model = CouncilModel(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable params: {n_params:,}")

    if args.no_pretrain:
        pretrain_path = None
        logger.info("SFT from random init (--no_pretrain specified)")
    elif args.pretrain_ckpt is not None:
        pretrain_path = args.pretrain_ckpt
    else:
        final_path = f"{config.checkpoint_dir}/pretrain_final.pt"
        if os.path.exists(final_path):
            pretrain_path = final_path
        else:
            step_ckpt, step_num = find_latest_pretrain_checkpoint(config.checkpoint_dir)
            pretrain_path = step_ckpt
            if step_ckpt is not None:
                logger.info(f"No pretrain_final.pt, using latest step checkpoint: {step_ckpt} (step {step_num})")

    if pretrain_path is not None and os.path.exists(pretrain_path):
        pretrain_state = torch.load(pretrain_path, map_location="cpu", weights_only=False)
        if isinstance(pretrain_state, dict) and "model" in pretrain_state:
            pretrain_state = pretrain_state["model"]
        pretrain_state = {
            k.replace("_orig_mod.", ""): v for k, v in pretrain_state.items()
        }
        missing, unexpected = model.load_state_dict(pretrain_state, strict=False)
        if missing:
            logger.info(f"Pretrain missing keys: {len(missing)}")
        if unexpected:
            logger.info(f"Pretrain unexpected keys: {len(unexpected)}")
        logger.info(f"Loaded pretrain weights from {pretrain_path}")
        del pretrain_state
    elif not args.no_pretrain:
        logger.info("No pretrain checkpoint found, SFT from random init")

    if config.use_lora:
        replaced = apply_lora(model, rank=config.lora_rank, alpha=config.lora_alpha)
        freeze_and_unfreeze_lora(model)
        trainable = count_trainable(model)
        logger.info(f"LoRA applied to {len(replaced)} layers, rank={config.lora_rank}, trainable={trainable:,}")

    # ==== RESUME TỪ CHECKPOINT MỚI NHẤT ====
    resume_ckpt, resume_step = find_latest_checkpoint(config.checkpoint_dir, use_lora=config.use_lora)
    resume_state = None
    if resume_ckpt is not None:
        logger.info(f"Found checkpoint: {resume_ckpt} (step {resume_step})")
        resume_state = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        # Strip _orig_mod. nếu có
        resume_state["model"] = {
            k.replace("_orig_mod.", ""): v for k, v in resume_state["model"].items()
        } if "model" in resume_state else None

    # Thử compile, có fallback
    use_compile = os.environ.get("USE_COMPILE", "1") == "1"
    if use_compile:
        try:
            logger.info("Compiling model (mode=default)...")
            model = torch.compile(model, mode="default", fullgraph=False, dynamic=True)
            logger.info("Compile OK")
        except Exception as e:
            logger.info(f"Compile failed, fallback eager: {e}")
            use_compile = False
    if config.use_lora:
        optimizer = bnb.optim.PagedAdamW8bit(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.sft_lr, betas=(0.9, 0.95)
        )
    else:
        optimizer = bnb.optim.PagedAdamW8bit(model.parameters(), lr=config.sft_lr, betas=(0.9, 0.95))

    total_steps = max(10000, config.sft_total_steps)
    save_interval = max(1, int(total_steps * config.save_every_ratio))
    # SỬA LỖI 5: Warmup + Cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: warmup_cosine(s, config.sft_warmup, total_steps)
    )

    model.train()
    optimizer.zero_grad()
    step = 0
    rec_window = []
    sample_start_step = 200
    sample_interval = max(config.sample_every_n_steps, 200)
    last_probe_optimal_N = 0

    # ==== LOAD RESUME STATE ====
    if resume_state is not None:
        try:
            base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            base_model.load_state_dict(resume_state["model"], strict=False)
            optimizer.load_state_dict(resume_state["optimizer"])
            scheduler.load_state_dict(resume_state["scheduler"])
            step = resume_state["step"]
            logger.info(f"Resumed from step {step}")
        except Exception as e:
            logger.info(f"Resume failed: {e}. Starting from scratch.")
    # ============================

    for epoch in range(config.sft_epochs):
        pbar = tqdm(dataloader, desc=f"SFT Epoch {epoch}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            sft_recursion = getattr(config, "sft_fixed_recursion", 2)
            with autocast(dtype=torch.bfloat16):
                _, loss = model(input_ids, labels=labels, num_recursion=sft_recursion, force_full_recursion=True)
                loss = loss / config.sft_grad_accum

            loss.backward()

            if (step + 1) % config.sft_grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            step += 1
            current_loss = loss.item() * config.sft_grad_accum
            current_lr = scheduler.get_last_lr()[0]

            base_for_rec = model._orig_mod if hasattr(model, "_orig_mod") else model
            rec_used = getattr(base_for_rec, "_last_would_stop_at", config.max_recursion)
            n_actual = getattr(base_for_rec, "_last_n_rec_used", config.max_recursion)
            rec_window.append(rec_used)

            if step % 10 == 0:
                avg_rec = sum(rec_window) / len(rec_window)
                pbar.set_postfix({
                    "loss": f"{current_loss:.4f}",
                    "lr": f"{current_lr:.2e}",
                    "rec": f"{sft_recursion}",
                    "avg_rec": f"{avg_rec:.2f}",
                })
                logger.log_step(step, {
                    "loss": current_loss,
                    "lr": current_lr,
                    "avg_rec": avg_rec,
                    "n_actual": n_actual,
                    "rec": sft_recursion,
                })
                rec_window.clear()
            else:
                pbar.set_postfix({
                    "loss": f"{current_loss:.4f}",
                    "lr": f"{current_lr:.2e}",
                })

            if step >= sample_start_step and step % sample_interval == 0:
                prompt_text, completion_text = sample_batch_output(
                    model, tokenizer, batch, device, max_new_tokens=64, temperature=0.8
                )
                if prompt_text is not None:
                    logger.log_sample(step, prompt_text, completion_text)
                else:
                    logger.info(f"[sample step {step}] FAILED: {completion_text}")

            if step % save_interval == 0 and step > 0:
                os.makedirs(config.checkpoint_dir, exist_ok=True)
                save_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                if config.use_lora:
                    ckpt_name = f"sft_lora_step_{step}.pt"
                else:
                    ckpt_name = f"sft_step_{step}.pt"
                torch.save({
                    "model": save_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step,
                    "best_N": last_probe_optimal_N,
                    "config": config,
                    "is_lora": config.use_lora,
                }, f"{config.checkpoint_dir}/{ckpt_name}")
                pct = step / total_steps * 100
                logger.info(f"Saved SFT checkpoint at step {step} ({pct:.1f}%) -> {ckpt_name}")
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    final_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    if config.use_lora:
        lora_final = f"{config.checkpoint_dir}/sft_lora_final.pt"
        torch.save({
            "model": final_model.state_dict(),
            "best_N": last_probe_optimal_N,
            "config": config,
            "is_lora": True,
        }, lora_final)
        logger.info(f"SFT LoRA completed. LoRA weights saved to {lora_final}")
        logger.info(f"Run `python main.py merge --lora_ckpt {lora_final}` to merge into base model.")
    else:
        torch.save({
            "model": final_model.state_dict(),
            "best_N": last_probe_optimal_N,
            "config": config,
        }, f"{config.checkpoint_dir}/sft_final.pt")
        logger.info("SFT completed.")
    return model

if __name__ == "__main__":
    import sys
    train_sft(sys.argv[1:])