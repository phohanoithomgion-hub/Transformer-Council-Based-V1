import os
import math
from dataclasses import dataclass

@dataclass
class CouncilConfig:
    d_model_backbone: int = 240
    num_blocks: int = 7
    num_streams: int = 16
    sparse_num_blocks: int = 16
    sparse_top_k: int = 6
    vocab_size: int = 50257
    d_model_thought: int = 384
    ffn_ratio: int = 4
    engram_table_size: int = 50000
    engram_mem_dim: int = 64
    max_recursion: int = 1
    max_seq_len: int = 256
    pretrain_max_tokens_per_packed_batch: int = 256
    sliding_window: int = 256
    fixed_recursion: int = 1
    adaptive_inference: bool = False

    global_num_heads: int = 8
    global_num_kv_heads: int = 2
    thought_num_heads: int = 16
    thought_num_kv_heads: int = 4
    head_dim: int = 80

    adv_top_k_a: int = 4
    adv_top_k_b: int = 4
    thought_head_top_k: int = 8

    mamba_d_state: int = 512
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_headdim: int = 64

    hyper_top_k: int = 4

    hybrid_reeval_interval: int = 128
    hybrid_max_full_extra: int = 2

    sft_batch_size: int = 16
    sft_grad_accum: int = 8
    sft_lr: float = 1e-4
    sft_warmup: int = 2000
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    sft_epochs: int = 1
    sft_max_tokens: int = 50_000_000_000
    sft_total_steps: int = 20000
    save_every_ratio: float = 1
    sft_fixed_recursion: int = 2

    gradient_clip_norm: float = 1.0

    backbone_checkpoint: bool = False
    attention_checkpoint: bool = True

    pretrain_batch_size: int = 32
    pretrain_grad_accum: int = 10
    pretrain_lr: float = 5e-4
    pretrain_lr_warmdown_steps: int = 50
    pretrain_warmup: int = 500
    pretrain_max_tokens: int = 30_000_000_000_000
    pretrain_max_steps: int = 10_000_000
    pretrain_save_every_ratio: float = 0.001
    pretrain_use_packing: bool = False
    pretrain_beta2: float = 0.9871
    pretrain_dataset_configs: list = None
    pretrain_num_recursion: int = 1
    pretrain_curriculum_phases: list = None
    pretrain_steps_per_phase: list = None

    rl_batch_size: int = 8
    rl_grad_accum: int = 16
    rl_lr: float = 2e-5
    rl_epochs: int = 1
    rl_max_tokens: int = 50_000_000_000
    rl_total_steps: int = 90000
    rl_clip_epsilon: float = 0.2
    rl_beta: float = 0.01
    rl_max_new_tokens: int = 256
    sample_every_n_steps: int = 500
    rl_temperature: float = 0.8
    rl_generate_recursion: int = 2
    rl_force_recursion: int = 2
    rl_format_weight: float = 0.3
    rl_accuracy_weight: float = 1.0
    rl_log_every_n_samples: int = 10
    rl_thinking_min_chars: int = 20
    use_cache: bool = True
    mamba_ssm_cache_dtype: str = "bfloat16"
    mamba_conv_cache_dtype: str = "bfloat16"

    oppo_beta0_prior: float = 0.5
    oppo_advantage_scale: float = 4.0

    aux_cont_loss_weight: float = 0.05

    max_samples_per_dataset: int = 2_000_000_000
    num_workers: int = 0

    device: str = "cuda"
    dtype: str = "bfloat16"
    seed: int = 42

    inference_repetition_penalty: float = 1.2
    inference_no_repeat_ngram_size: int = 3

    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"


def get_config():
    config = CouncilConfig()
    if config.pretrain_dataset_configs is None:
        config.pretrain_dataset_configs = [
            "xerv_pure",
            "automathtext-v2-ultra",
            "fineweb-edu",
            "fineweb",
            "dclm-baseline",
        ]
    return config