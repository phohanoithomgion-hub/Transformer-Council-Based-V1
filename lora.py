import math
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base_layer, rank=16, alpha=32):
        super().__init__()
        self.base = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(rank, in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        base_out = self.base(x)
        lora_out = (x @ self.lora_A.t()) @ self.lora_B.t()
        return base_out + lora_out * self.scaling

    def merge_and_unload(self):
        with torch.no_grad():
            self.base.weight.data.add_(self.scaling * (self.lora_B @ self.lora_A))
        return self.base

LORA_TARGET_PATTERNS = (
    "logic.t1_experts.", "logic.router_t1", "logic.hyper_a", "logic.hyper_b", "logic.shared_expert",
    "dynamic.backbone.", "multidim.euclid.", "multidim.hyper.", "multidim.sph.", "multidim.gate.", "multidim.out_proj",
    "quantum.amp_encoder.", "quantum.measurement",
    "decode_layer.decode.",
    "backbone.attn_layers.", "engram.mem_proj", "engram.gate", "proj_in", "proj_final",
)

def _should_target(name):
    for p in LORA_TARGET_PATTERNS:
        if name == p or name.startswith(p):
            return True
    return False

def apply_lora(model, rank=16, alpha=32):
    replaced = []
    modules_dict = dict(model.named_modules())
    for name, module in list(modules_dict.items()):
        if isinstance(module, nn.Linear) and _should_target(name):
            if "." not in name:
                continue
            parent_name, child_name = name.rsplit(".", 1)
            parent = modules_dict[parent_name]
            lora = LoRALinear(module, rank=rank, alpha=alpha)
            setattr(parent, child_name, lora)
            replaced.append(name)
    return replaced

def freeze_and_unfreeze_lora(model):
    for p in model.parameters():
        p.requires_grad = False
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True
    for name, p in model.named_parameters():
        if name.endswith(".alpha") or name.endswith(".gamma"):
            p.requires_grad = True
        if "decision.decision" in name and name.endswith(".bias"):
            p.requires_grad = True

def merge_lora(model):
    modules_dict = dict(model.named_modules())
    for name, module in list(modules_dict.items()):
        if isinstance(module, LoRALinear):
            if "." not in name:
                continue
            parent_name, child_name = name.rsplit(".", 1)
            parent = modules_dict[parent_name]
            merged = module.merge_and_unload()
            setattr(parent, child_name, merged)

def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)