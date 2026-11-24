import os
import sys
import math
import torch
import torch.nn.functional as F
import bitsandbytes as bnb
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import CouncilModel
from dataset import StreamingPretrainDataset, PackedPretrainDataset, get_tokenizer
from config import get_config
from log import TrainingLogger


_CRASH_SAVE_FN = None


def _install_crash_handler():
    _orig = sys.excepthook

    def _hook(exc_type, exc_val, exc_tb):
        _orig(exc_type, exc_val, exc_tb)
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        if _CRASH_SAVE_FN is not None:
            try:
                _CRASH_SAVE_FN()
            except Exception:
                pass
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except Exception:
                pass
        os._exit(1)

    sys.excepthook = _hook


def _register_crash_save(fn):
    global _CRASH_SAVE_FN
    _CRASH_SAVE_FN = fn


def set_vram_budget(fraction=0.9):
    if not torch.cuda.is_available():
        return
    total = torch.cuda.get_device_properties(0).total_memory
    budget_bytes = int(total * fraction)
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    print(f"[vram] total={total/1e9:.2f}GB budget={budget_bytes/1e9:.2f}GB fraction={fraction}")


def find_latest_pretrain_checkpoint(ckpt_dir):
    import glob
    candidates = (
        glob.glob(os.path.join(ckpt_dir, "pretrain_curriculum_*.pt")) +
        glob.glob(os.path.join(ckpt_dir, "pretrain_step_*.pt"))
    )
    if not candidates:
        return None, 0
    def get_step(path):
        try:
            return int(os.path.basename(path).split("_")[-1].split(".")[0])
        except Exception:
            return 0
    candidates.sort(key=get_step, reverse=True)
    return candidates[0], get_step(candidates[0])


def warmup_cosine(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))


def custom_collate_fn(batch):
    max_cu_len = max(len(x["cu_seqlens"]) for x in batch)
    for x in batch:
        cu = x["cu_seqlens"]
        pad_len = max_cu_len - len(cu)
        if pad_len > 0:
            x["cu_seqlens"] = torch.cat([cu, torch.full((pad_len,), cu[-1], dtype=cu.dtype)])
    return torch.utils.data.dataloader.default_collate(batch)


def _divide_grads_by(model, denom):
    if denom <= 0:
        return
    for p in model.parameters():
        if p.grad is not None:
            p.grad.div_(denom)


def train_pretrain(args_list=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_phase", type=int, default=None)
    args = parser.parse_args(args_list or [])

    _install_crash_handler()

    config = get_config()
    device = torch.device(config.device)
    set_vram_budget(fraction=0.90)
    torch.manual_seed(config.seed)

    tokenizer = get_tokenizer()
    config.vocab_size = len(tokenizer)
    config.eos_token_id = tokenizer.eos_token_id
    print(f"Tokenizer vocab_size = {config.vocab_size}")

    logger = TrainingLogger(config.log_dir, "pretrain")
    logger.info(f"Config: {config}")

    n_rec = getattr(config, "pretrain_num_recursion", 1)
    grad_accum = config.pretrain_grad_accum
    total_optim_steps = config.pretrain_max_steps
    max_tokens = config.pretrain_max_tokens
    num_workers = 0
    use_packing = getattr(config, "pretrain_use_packing", False)
    save_interval = max(1, int(total_optim_steps * config.pretrain_save_every_ratio))
    logger.info(
        f"Single-phase pretrain: n_rec={n_rec}, grad_accum={grad_accum}, "
        f"total_optim_steps={total_optim_steps}, warmup={config.pretrain_warmup}, "
        f"max_tokens={max_tokens/1e12:.2f}T, num_workers={num_workers}, packing={use_packing}"
    )

    config.backbone_checkpoint = True
    model = CouncilModel(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable params: {n_params:,}")

    resume_ckpt, _ = find_latest_pretrain_checkpoint(config.checkpoint_dir)
    resume_state = None
    optim_step = 0
    micro_step = 0
    total_real_tokens = 0
    total_input_tokens = 0
    total_samples_consumed = 0
    total_packs_consumed = 0
    dataset_state = None
    if resume_ckpt is not None:
        logger.info(f"Found pretrain checkpoint: {resume_ckpt}")
        resume_state = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        resume_state["model"] = {
            k.replace("_orig_mod.", ""): v
            for k, v in resume_state["model"].items()
        }
        missing, unexpected = model.load_state_dict(resume_state["model"], strict=False)
        logger.info(f"Resumed model weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        optim_step = resume_state.get("step", resume_state.get("total_step", 0))
        micro_step = resume_state.get("micro_step", optim_step * grad_accum)
        total_real_tokens = int(resume_state.get("total_real_tokens", 0))
        total_input_tokens = int(resume_state.get("total_input_tokens", total_real_tokens))
        dataset_state = resume_state.get("dataset_state", None)

        old_bs = getattr(resume_state.get("config"), "pretrain_batch_size", None)

        if use_packing:
            total_packs_consumed = int(resume_state.get("total_packs_consumed", micro_step))
            total_samples_consumed = int(resume_state.get("total_samples_consumed", 0))
            if old_bs is not None and old_bs != config.pretrain_batch_size:
                logger.warning(
                    f"pretrain_batch_size changed ({old_bs} -> {config.pretrain_batch_size}); "
                    f"packed-mode resume will be approximate"
                )
        else:
            if "total_samples_consumed" in resume_state:
                total_samples_consumed = int(resume_state["total_samples_consumed"])
            else:
                old_bs_eff = old_bs if old_bs is not None else config.pretrain_batch_size
                total_samples_consumed = micro_step * old_bs_eff
                logger.warning(
                    f"Old checkpoint lacks total_samples_consumed; "
                    f"estimated {total_samples_consumed} (micro_step={micro_step} * old_bs={old_bs_eff})"
                )

        logger.info(
            f"Resumed optim_step={optim_step}, micro_step={micro_step}, "
            f"loss_tokens={total_real_tokens/1e9:.3f}B, input_tokens={total_input_tokens/1e9:.3f}B, "
            f"samples_consumed={total_samples_consumed}, packs_consumed={total_packs_consumed}"
        )

    if optim_step >= total_optim_steps or total_real_tokens >= max_tokens:
        logger.info(
            f"Already done: optim_step={optim_step}/{total_optim_steps}, "
            f"loss_tokens={total_real_tokens/1e9:.3f}B/{max_tokens/1e9:.1f}B"
        )
        return model

    if use_packing:
        base_dataset = PackedPretrainDataset(
            tokenizer,
            max_seq_len=config.max_seq_len,
            max_tokens_per_batch=config.pretrain_max_tokens_per_packed_batch,
            config_names=config.pretrain_dataset_configs,
        )
        print(f"[packing] enabled, max_tokens={config.pretrain_max_tokens_per_packed_batch}")
    else:
        base_dataset = StreamingPretrainDataset(
            tokenizer,
            max_seq_len=config.max_seq_len,
            config_names=config.pretrain_dataset_configs,
        )

    skip_batches = 0
    if num_workers == 0 and dataset_state is not None and hasattr(base_dataset, "load_state_dict"):
        try:
            base_dataset.load_state_dict(dataset_state)
            skip_batches = 0
            logger.info("Using dataset state_dict for fast resume (num_workers=0)")
        except Exception as e:
            logger.warning(f"dataset.load_state_dict failed: {e}; falling back to islice")
            skip_batches = _compute_skip(use_packing, total_packs_consumed, total_samples_consumed, config, logger)
    elif resume_state is not None:
        skip_batches = _compute_skip(use_packing, total_packs_consumed, total_samples_consumed, config, logger)

    if num_workers > 1:
        from torch.utils.data import IterableDataset
        class WorkerShardedDataset(IterableDataset):
            def __init__(self, base):
                super().__init__()
                self.base = base
            def __iter__(self):
                from torch.utils.data import get_worker_info
                info = get_worker_info()
                if info is None or info.num_workers <= 1:
                    yield from self.base
                    return
                if hasattr(self.base, "shard"):
                    yield from self.base.shard(info.num_workers, info.id)
                    return
                for i, ex in enumerate(self.base):
                    if i % info.num_workers == info.id:
                        yield ex
            def state_dict(self):
                return self.base.state_dict() if hasattr(self.base, "state_dict") else None
            def load_state_dict(self, state):
                if hasattr(self.base, "load_state_dict"):
                    return self.base.load_state_dict(state)
        dataset = WorkerShardedDataset(base_dataset)
    else:
        dataset = base_dataset

    dataloader = DataLoader(
        dataset,
        batch_size=config.pretrain_batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        multiprocessing_context="spawn" if num_workers > 0 else None,
        collate_fn=custom_collate_fn if use_packing else None,
    )

    if skip_batches > 0:
        logger.info(f"Skipping {skip_batches} batches via islice (slow but correct)")
        from itertools import islice
        dataloader = islice(dataloader, skip_batches, None)

    use_compile = os.environ.get("USE_COMPILE", "1") == "1"
    if use_compile:
        try:
            logger.info("Compiling model...")
            model = torch.compile(model, mode="default", fullgraph=False, dynamic=True)
            logger.info("Compile OK")
        except Exception as e:
            logger.info(f"Compile failed: {e}")

    beta2 = getattr(config, "pretrain_beta2", 0.99)
    optimizer = bnb.optim.AdamW8bit(
        model.parameters(), lr=config.pretrain_lr, betas=(0.9, beta2)
    )
    print(f"[optimizer] beta2={beta2}")
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: warmup_cosine(s, config.pretrain_warmup, total_optim_steps),
    )
    if resume_state is not None:
        try:
            if "optimizer" in resume_state:
                optimizer.load_state_dict(resume_state["optimizer"])
            if "scheduler" in resume_state:
                scheduler.load_state_dict(resume_state["scheduler"])
        except Exception as e:
            logger.info(f"Optimizer resume failed: {e}")
            for pg in optimizer.param_groups:
                pg['lr'] = config.pretrain_lr

    def _save_on_crash():
        try:
            save_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            ds_state = None
            try:
                if num_workers == 0 and hasattr(base_dataset, "state_dict"):
                    ds_state = base_dataset.state_dict()
            except Exception:
                pass
            torch.save({
                "model": save_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": optim_step,
                "micro_step": micro_step,
                "total_real_tokens": total_real_tokens,
                "total_input_tokens": total_input_tokens,
                "total_samples_consumed": total_samples_consumed,
                "total_packs_consumed": total_packs_consumed,
                "dataset_state": ds_state,
                "config": config,
                "stop_reason": "crash",
            }, f"{config.checkpoint_dir}/pretrain_crash_{optim_step}.pt")
        except Exception:
            pass

    _register_crash_save(_save_on_crash)
    model.train()
    optimizer.zero_grad()

    accum_valid = 0.0
    accum_loss_sum = 0.0
    accum_micro = 0

    stop_reason = None
    total_micro_remaining = max(1, total_optim_steps * grad_accum - micro_step)
    pbar = tqdm(dataloader, desc=f"Pretrain n_rec={n_rec}", total=total_micro_remaining)

    for batch in pbar:
        if optim_step >= total_optim_steps:
            stop_reason = "steps"
            break
        if total_real_tokens >= max_tokens:
            stop_reason = "tokens"
            if accum_micro > 0 and accum_valid > 0:
                _divide_grads_by(model, accum_valid)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optim_step += 1
                micro_step += accum_micro
                accum_micro = 0
                accum_valid = 0.0
                accum_loss_sum = 0.0
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        n_input = int(batch["n_tokens"].sum().item())
        total_input_tokens += n_input

        n_samples_batch = int(batch.get("n_samples", input_ids.size(0))) if use_packing else input_ids.size(0)
        total_samples_consumed += n_samples_batch
        if use_packing:
            total_packs_consumed += input_ids.size(0) // config.pretrain_max_tokens_per_packed_batch

        cu_seqlens = batch.get("cu_seqlens", None)
        if cu_seqlens is not None:
            cu_seqlens = cu_seqlens.to(device)
            if cu_seqlens.dim() == 1:
                cu_seqlens = cu_seqlens.unsqueeze(0)

        with autocast("cuda", dtype=torch.bfloat16):
            _, loss_mean, n_valid = model(
                input_ids,
                labels=labels,
                num_recursion=n_rec,
                force_full_recursion=True,
                cu_seqlens=cu_seqlens,
                return_loss_info=True,
            )
        loss_sum = loss_mean.float() * n_valid.float()
        loss_sum.backward()

        accum_valid += float(n_valid.item())
        accum_loss_sum += float(loss_sum.item())
        accum_micro += 1
        total_real_tokens += int(n_valid.item())

        do_step = (accum_micro >= grad_accum)
        is_last_steps = (optim_step + 1 >= total_optim_steps)
        is_last_tokens = (total_real_tokens >= max_tokens)
        is_last = is_last_steps or is_last_tokens

        if do_step or is_last:
            step_loss = accum_loss_sum / max(accum_valid, 1.0)
            _divide_grads_by(model, accum_valid)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optim_step += 1
            micro_step += accum_micro
            accum_valid = 0.0
            accum_loss_sum = 0.0
            accum_micro = 0

            if optim_step % 10 == 0:
                last_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({
                    "opt_step": optim_step,
                    "loss": f"{step_loss:.4f}",
                    "lr": f"{last_lr:.2e}",
                    "loss_tok": f"{total_real_tokens/1e9:.2f}B",
                    "in_tok": f"{total_input_tokens/1e9:.2f}B",
                    "samples": f"{total_samples_consumed/1e6:.2f}M",
                    "pct": f"{optim_step/total_optim_steps*100:.1f}%",
                })
                logger.log_step(optim_step, {
                    "loss": step_loss,
                    "lr": last_lr,
                    "loss_tokens_B": total_real_tokens / 1e9,
                    "input_tokens_B": total_input_tokens / 1e9,
                    "samples_consumed": total_samples_consumed,
                    "n_rec": n_rec,
                    "micro_step": micro_step,
                })

            if optim_step % 100 == 0:
                import gc
                gc.collect()

            if optim_step % save_interval == 0:
                save_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                os.makedirs(config.checkpoint_dir, exist_ok=True)
                ds_state = None
                try:
                    if num_workers == 0 and hasattr(base_dataset, "state_dict"):
                        ds_state = base_dataset.state_dict()
                except Exception as e:
                    logger.warning(f"dataset.state_dict failed at save: {e}")
                torch.save({
                    "model": save_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": optim_step,
                    "micro_step": micro_step,
                    "total_real_tokens": total_real_tokens,
                    "total_input_tokens": total_input_tokens,
                    "total_samples_consumed": total_samples_consumed,
                    "total_packs_consumed": total_packs_consumed,
                    "dataset_state": ds_state,
                    "config": config,
                }, f"{config.checkpoint_dir}/pretrain_step_{optim_step}.pt")
                logger.info(f"Saved pretrain_step_{optim_step}.pt (optim_step={optim_step}, samples={total_samples_consumed})")

            if is_last:
                if is_last_steps:
                    stop_reason = "steps"
                elif is_last_tokens:
                    stop_reason = "tokens"
                logger.info(
                    f"Stopping after step {optim_step}: reason={stop_reason}, "
                    f"loss_tokens={total_real_tokens/1e9:.3f}B, input_tokens={total_input_tokens/1e9:.3f}B, "
                    f"samples={total_samples_consumed/1e6:.2f}M"
                )
                break

    if accum_micro > 0 and accum_valid > 0:
        _divide_grads_by(model, accum_valid)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optim_step += 1
        micro_step += accum_micro
        accum_micro = 0

    if stop_reason is None:
        stop_reason = "dataset_exhausted"

    if optim_step == 0 and total_real_tokens == 0:
        logger.error(
            "No optimizer step and no tokens processed. Dataset may be empty or failed to load. "
            "NOT writing pretrain_final.pt."
        )
        raise RuntimeError(
            "Pretraining terminated without any training progress. "
            "Check dataset availability and __iter__ implementation."
        )

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    save_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    ds_state = None
    try:
        if num_workers == 0 and hasattr(base_dataset, "state_dict"):
            ds_state = base_dataset.state_dict()
    except Exception:
        pass

    final_path = f"{config.checkpoint_dir}/pretrain_final.pt"
    torch.save({
        "model": save_model.state_dict(),
        "step": optim_step,
        "micro_step": micro_step,
        "total_real_tokens": total_real_tokens,
        "total_input_tokens": total_input_tokens,
        "total_samples_consumed": total_samples_consumed,
        "total_packs_consumed": total_packs_consumed,
        "dataset_state": ds_state,
        "stop_reason": stop_reason,
        "config": config,
    }, final_path)

    if stop_reason in ("steps", "tokens"):
        logger.info(
            f"Pretrain COMPLETED ({stop_reason}). "
            f"optim_step={optim_step}/{total_optim_steps}, "
            f"loss_tokens={total_real_tokens/1e9:.3f}B/{max_tokens/1e9:.1f}B, "
            f"input_tokens={total_input_tokens/1e9:.3f}B, samples={total_samples_consumed/1e6:.2f}M"
        )
    else:
        logger.warning(
            f"Pretrain ENDED EARLY ({stop_reason}). "
            f"optim_step={optim_step}/{total_optim_steps}, "
            f"loss_tokens={total_real_tokens/1e9:.3f}B/{max_tokens/1e9:.1f}B, "
            f"input_tokens={total_input_tokens/1e9:.3f}B, samples={total_samples_consumed/1e6:.2f}M. "
            f"Check dataset streams."
        )
        logger.warning(f"Final checkpoint saved as {final_path} (early-stop)")
    return model


def _compute_skip(use_packing, total_packs_consumed, total_samples_consumed, config, logger):
    if use_packing:
        logger.info(f"Resume packed mode: skip {total_packs_consumed} packs (= {total_packs_consumed // config.pretrain_batch_size} batches + {total_packs_consumed % config.pretrain_batch_size} packs)")
        return total_packs_consumed // config.pretrain_batch_size
    bs = config.pretrain_batch_size
    skip = total_samples_consumed // bs
    remainder = total_samples_consumed % bs
    if remainder != 0:
        logger.warning(
            f"total_samples_consumed={total_samples_consumed} not divisible by batch_size={bs} "
            f"(remainder={remainder}); will skip {skip} batches, losing {remainder} samples precision"
        )
    logger.info(f"Resume non-packed: skip {skip} batches ({skip * bs} samples)")
    return skip


if __name__ == "__main__":
    train_pretrain()