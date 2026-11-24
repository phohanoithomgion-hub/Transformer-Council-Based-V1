import inspect
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import torch._dynamo

torch._dynamo.config.recompile_limit = 16
torch._dynamo.config.cache_size_limit = 32
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

def _detect_sdpa_gqa():
    sdpa = getattr(F, "scaled_dot_product_attention", None)
    if sdpa is None:
        return False
    try:
        sig = inspect.signature(sdpa)
        if "enable_gqa" in sig.parameters:
            return True
        if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return False
    except (ValueError, TypeError):
        pass
    try:
        with torch.no_grad():
            q = torch.zeros((1, 2, 2, 8), device="cpu", dtype=torch.float32)
            k = torch.zeros((1, 1, 2, 8), device="cpu", dtype=torch.float32)
            v = torch.zeros((1, 1, 2, 8), device="cpu", dtype=torch.float32)
            sdpa(q, k, v, is_causal=True, enable_gqa=True)
        return True
    except TypeError:
        return False
    except Exception:
        return "enable_gqa" in (getattr(sdpa, "__doc__", None) or "")


_HAS_SDPA_GQA = _detect_sdpa_gqa()

try:
    import mamba_ssm.modules.mamba2 as _mamba2_mod
    from mamba_ssm.modules.mamba2 import Mamba2 as _Mamba2Native
    _HAS_MAMBA2 = True
    _HAS_CAUSAL_CONV1D_FN = getattr(_mamba2_mod, "causal_conv1d_fn", None) is not None
    _HAS_CAUSAL_CONV1D_VARLEN = getattr(_mamba2_mod, "causal_conv1d_varlen_states", None) is not None
except ImportError:
    _mamba2_mod = None
    _Mamba2Native = None
    _HAS_MAMBA2 = False
    _HAS_CAUSAL_CONV1D_FN = False
    _HAS_CAUSAL_CONV1D_VARLEN = False

try:
    from mamba_ssm.utils.generation import InferenceParams as _OfficialInferenceParams
except ImportError:
    _OfficialInferenceParams = None


def _str_to_dtype(s):
    if s is None:
        return None
    if isinstance(s, torch.dtype):
        return s
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    return mapping.get(str(s).lower(), None)


def _create_inference_params(max_seqlen, max_batch_size, num_heads, head_dim, n_layer):
    if _OfficialInferenceParams is not None:
        try:
            return _OfficialInferenceParams(
                max_seqlen=max_seqlen,
                max_batch_size=max_batch_size,
                num_heads=num_heads,
                head_dim=head_dim,
                n_layer=n_layer,
            )
        except TypeError:
            try:
                return _OfficialInferenceParams(
                    max_seqlen=max_seqlen,
                    max_batch_size=max_batch_size,
                )
            except TypeError:
                pass

    class _Fallback:
        pass

    fp = _Fallback()
    fp.max_seqlen = max_seqlen
    fp.max_batch_size = max_batch_size
    fp.seqlen_offset = 0
    fp.key_value_memory_dict = {}
    return fp


class CausalConv1d(nn.Module):
    def __init__(self, d_model, kernel_size=5):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, groups=d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def _build_cache_from_real(self, x, attention_mask):
        B, T, D = x.shape
        k = self.kernel_size
        new_cache = torch.zeros(B, k - 1, D, device=x.device, dtype=x.dtype)
        if k <= 1 or T == 0:
            return new_cache
        am_bool = attention_mask[:, :T].bool()
        lengths = am_bool.long().sum(dim=1)
        rank = am_bool.long().cumsum(dim=1)
        valid_src = am_bool & (rank > (lengths.unsqueeze(1) - (k - 1)))
        if valid_src.any():
            b_idx, t_idx = valid_src.nonzero(as_tuple=True)
            dst_idx = rank[b_idx, t_idx] - lengths[b_idx] + (k - 2)
            new_cache[b_idx, dst_idx] = x[b_idx, t_idx].detach()
        return new_cache

    def _build_seq_cache_from_real(self, seq_idx, attention_mask, T):
        B = seq_idx.shape[0]
        k = self.kernel_size
        new_seq_cache = torch.full((B, k - 1), -1, device=seq_idx.device, dtype=torch.int32)
        if k <= 1 or T == 0:
            return new_seq_cache
        seq_t = seq_idx[:, :T].to(torch.int32)
        if attention_mask is not None:
            am_bool = attention_mask[:, :T].bool()
            lengths = am_bool.long().sum(dim=1)
            rank = am_bool.long().cumsum(dim=1)
            valid_src = am_bool & (rank > (lengths.unsqueeze(1) - (k - 1)))
            if valid_src.any():
                b_idx, t_idx = valid_src.nonzero(as_tuple=True)
                dst_idx = rank[b_idx, t_idx] - lengths[b_idx] + (k - 2)
                new_seq_cache[b_idx, dst_idx] = seq_t[b_idx, t_idx]
        else:
            if T >= k - 1:
                new_seq_cache = seq_t[:, -(k - 1):].detach()
            elif T > 0:
                new_seq_cache[:, (k - 1 - T):] = seq_t.detach()
        return new_seq_cache

    def _run_conv_with_seq(self, x, seq_idx, T):
        if T == 0:
            return torch.zeros_like(x)
        k = self.kernel_size
        q_seq = seq_idx[:, :T].to(torch.int32)
        if k <= 1:
            valid = (q_seq >= 0).unsqueeze(-1).to(x.dtype)
            xc = self.conv(x.transpose(1, 2))[:, :, :T].transpose(1, 2)
            return xc * valid
        x_pad = F.pad(x, (0, 0, k - 1, 0))
        tap_seq = F.pad(q_seq, (k - 1, 0), value=-1)
        return self._run_conv_with_cache_seq(x_pad, tap_seq, q_seq, T)

    def _run_conv_with_cache_seq(self, cat_x, tap_seq, q_seq, T):
        B, _, D = cat_x.shape
        k = self.kernel_size
        xc_t = cat_x.transpose(1, 2)
        weight = self.conv.weight.to(cat_x.dtype)
        out = torch.zeros(B, D, T, device=cat_x.device, dtype=cat_x.dtype)
        for j in range(k):
            tap = xc_t[:, :, j:j + T]
            k_seq = tap_seq[:, j:j + T]
            m = (q_seq == k_seq) & (k_seq >= 0)
            out = out + weight[:, 0, j].view(1, D, 1) * tap * m.unsqueeze(1).to(tap.dtype)
        return out.transpose(1, 2)

    def forward(self, x, conv_cache=None, seq_idx=None, attention_mask=None,
                seq_cache=None, return_seq_cache=False):
        B, T, D = x.shape
        k = self.kernel_size
        new_seq_cache = None
        am_f_out = None
        if conv_cache is None:
            if attention_mask is not None and T > 0:
                if attention_mask.size(1) < T:
                    raise ValueError(f"attention_mask length {attention_mask.size(1)} < input length {T}")
                am_bool = attention_mask[:, :T].bool()
                am_f_out = am_bool.unsqueeze(-1).to(x.dtype)
                x = x * am_f_out
                residual = x
                real_pos = am_bool.long().cumsum(dim=1) - 1
                pos_safe = torch.where(am_bool, real_pos, torch.full_like(real_pos, T))
                x_comp = torch.zeros(B, T + 1, D, device=x.device, dtype=x.dtype).scatter(
                    1, pos_safe.unsqueeze(-1).expand(-1, -1, D), x)[:, :T, :]
                if seq_idx is not None:
                    seq_comp = torch.full((B, T + 1), -1, device=seq_idx.device, dtype=torch.int32).scatter(
                        1, pos_safe, seq_idx[:, :T].to(torch.int32))[:, :T]
                    xc_comp = self._run_conv_with_seq(x_comp, seq_comp, T)
                else:
                    xc_pad = F.pad(x_comp.transpose(1, 2), (k - 1, 0))
                    xc_comp = self.conv(xc_pad)[:, :, :T].transpose(1, 2)
                xc = F.pad(xc_comp, (0, 0, 0, 1)).gather(1, pos_safe.unsqueeze(-1).expand(-1, -1, D))
                xc = xc * am_f_out
                if k > 1:
                    lengths = am_bool.long().sum(dim=1)
                    cache_offs = lengths.unsqueeze(1) - (k - 1) + torch.arange(k - 1, device=x.device).unsqueeze(0)
                    valid_c = cache_offs >= 0
                    new_cache = x_comp.gather(
                        1, cache_offs.clamp(min=0, max=max(T - 1, 0)).unsqueeze(-1).expand(-1, -1, D)
                    ).detach() * valid_c.unsqueeze(-1).to(x.dtype)
                else:
                    new_cache = torch.zeros(B, 0, D, device=x.device, dtype=x.dtype)
            else:
                residual = x
                if seq_idx is not None:
                    valid_s = (seq_idx[:, :T] >= 0).unsqueeze(-1).to(x.dtype)
                    residual = x * valid_s
                    xc = self._run_conv_with_seq(residual, seq_idx[:, :T], T) * valid_s
                    am_f_out = valid_s
                else:
                    xc_pad = F.pad(x.transpose(1, 2), (k - 1, 0))
                    xc = self.conv(xc_pad)[:, :, :T].transpose(1, 2)
                if T >= k - 1:
                    new_cache = residual[:, -(k - 1):].detach()
                else:
                    pad = torch.zeros(B, k - 1 - T, D, device=x.device, dtype=x.dtype)
                    new_cache = torch.cat([pad, residual.detach()], dim=1)
            if return_seq_cache and seq_idx is not None:
                new_seq_cache = self._build_seq_cache_from_real(seq_idx, attention_mask, T)
        else:
            if attention_mask is not None and T > 0:
                if attention_mask.size(1) < T:
                    raise ValueError(f"attention_mask length {attention_mask.size(1)} < chunk length {T}")
                am_cur = attention_mask[:, -T:].bool()
                am_f_out = am_cur.unsqueeze(-1).to(x.dtype)
                x = x * am_f_out
                residual = x
                if T == 1:
                    cat_x = torch.cat([conv_cache, x], dim=1)
                    if seq_cache is not None and seq_idx is not None:
                        q_seq = seq_idx[:, :1].to(torch.int32)
                        tap_seq = torch.cat([seq_cache.to(torch.int32), q_seq], dim=1)
                        valid_taps = (tap_seq == q_seq) & (tap_seq >= 0)
                        xc_in = (cat_x * valid_taps.unsqueeze(-1).to(x.dtype)).transpose(1, 2)
                        xc = self.conv(xc_in)[:, :, -1:].transpose(1, 2)
                        if return_seq_cache:
                            new_seq_cand = tap_seq[:, -(k - 1):].detach() if k > 1 else tap_seq[:, :0]
                            new_seq_cache = torch.where(am_cur, new_seq_cand, seq_cache)
                    else:
                        xc = self.conv(cat_x.transpose(1, 2))[:, :, -1:].transpose(1, 2)
                    xc = xc * am_f_out
                    if k > 1:
                        new_cache_cand = cat_x[:, -(k - 1):].detach()
                        new_cache = torch.where(am_cur.unsqueeze(-1), new_cache_cand, conv_cache)
                    else:
                        new_cache = cat_x[:, :0].detach()
                else:
                    real_pos = am_cur.long().cumsum(dim=1) - 1
                    pos_safe = torch.where(am_cur, real_pos, torch.full_like(real_pos, T))
                    x_comp = torch.zeros(B, T + 1, D, device=x.device, dtype=x.dtype).scatter(
                        1, pos_safe.unsqueeze(-1).expand(-1, -1, D), x)[:, :T, :]
                    cat_comp = torch.cat([conv_cache, x_comp], dim=1)
                    if seq_cache is not None and seq_idx is not None:
                        seq_in = seq_idx[:, :T].to(torch.int32)
                        seq_comp = torch.full((B, T + 1), -1, device=seq_idx.device, dtype=torch.int32).scatter(
                            1, pos_safe, seq_in)[:, :T]
                        tap_seq = torch.cat([seq_cache.to(torch.int32), seq_comp], dim=1)
                        xc_comp = self._run_conv_with_cache_seq(cat_comp, tap_seq, seq_comp, T)
                        if return_seq_cache:
                            if k > 1:
                                lengths = am_cur.long().sum(dim=1)
                                cache_offs = lengths.unsqueeze(1) + torch.arange(k - 1, device=x.device).unsqueeze(0)
                                new_seq_cache = tap_seq.gather(1, cache_offs.clamp(min=0, max=k - 2 + T)).detach()
                            else:
                                new_seq_cache = tap_seq[:, :0]
                    else:
                        xc_comp = self.conv(cat_comp.transpose(1, 2))[:, :, -T:].transpose(1, 2)
                    xc = F.pad(xc_comp, (0, 0, 0, 1)).gather(1, pos_safe.unsqueeze(-1).expand(-1, -1, D))
                    xc = xc * am_f_out
                    if k > 1:
                        lengths = am_cur.long().sum(dim=1)
                        cache_offs = lengths.unsqueeze(1) + torch.arange(k - 1, device=x.device).unsqueeze(0)
                        new_cache = cat_comp.gather(
                            1, cache_offs.clamp(min=0, max=k - 2 + T).unsqueeze(-1).expand(-1, -1, D)
                        ).detach()
                    else:
                        new_cache = torch.zeros(B, 0, D, device=x.device, dtype=x.dtype)
            else:
                residual = x
                cat_x = torch.cat([conv_cache, x], dim=1)
                if seq_cache is not None and seq_idx is not None:
                    q_seq = seq_idx[:, :T].to(torch.int32)
                    tap_seq = torch.cat([seq_cache.to(torch.int32), q_seq], dim=1)
                    if T == 1:
                        valid_taps = (tap_seq == q_seq) & (tap_seq >= 0)
                        xc_in = (cat_x * valid_taps.unsqueeze(-1).to(x.dtype)).transpose(1, 2)
                        xc = self.conv(xc_in)[:, :, -1:].transpose(1, 2)
                    else:
                        xc = self._run_conv_with_cache_seq(cat_x, tap_seq, q_seq, T)
                    if return_seq_cache:
                        new_seq_cache = tap_seq[:, -(k - 1):].detach() if k > 1 else tap_seq[:, :0]
                else:
                    xc = self.conv(cat_x.transpose(1, 2))[:, :, -T:].transpose(1, 2)
                new_cache = cat_x[:, -(k - 1):].detach() if k > 1 else cat_x[:, :0].detach()
        xc = self.proj(xc) * F.silu(self.gate(xc))
        out = self.norm(residual + xc)
        if am_f_out is not None:
            out = out * am_f_out
        if return_seq_cache:
            return out, new_cache, new_seq_cache
        return out, new_cache


def _causal_mask(T_q, T_kv, past_len, device, window_size=None):
    q_pos = torch.arange(past_len, past_len + T_q, device=device)
    k_pos = torch.arange(T_kv, device=device)
    mask = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
    if window_size is not None:
        win = (q_pos.unsqueeze(1) - k_pos.unsqueeze(0)) <= window_size
        mask = mask & win
    return mask


def _seq_boundary_mask(seq_idx, T_q, T_kv, past_len, device, past_seq_idx=None):
    if seq_idx is None:
        return None
    if past_len == 0:
        if seq_idx.size(1) < T_q or seq_idx.size(1) < T_kv:
            return None
        q_seq = seq_idx[:, :T_q]
        k_seq = seq_idx[:, :T_kv]
    else:
        if past_seq_idx is None or seq_idx.size(1) < T_q:
            return None
        q_seq = seq_idx[:, :T_q]
        k_seq = torch.cat([past_seq_idx, q_seq], dim=1)
        if k_seq.size(1) != T_kv:
            return None
    same = (q_seq.unsqueeze(-1) == k_seq.unsqueeze(-2)) & (q_seq.unsqueeze(-1) >= 0) & (k_seq.unsqueeze(-2) >= 0)
    return same.unsqueeze(1)


def _safe_sdpa(q, k, v, attn_mask, mutate_mask=False):
    if attn_mask.size(-1) == 0:
        return torch.zeros_like(q)
    has_valid = attn_mask.any(dim=-1, keepdim=True)
    safe_mask = attn_mask if mutate_mask else attn_mask.clone()
    safe_mask[..., 0] = safe_mask[..., 0] | (~has_valid.squeeze(-1))
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=safe_mask, is_causal=False)
    return torch.where(has_valid, out, torch.zeros_like(out))


class GQAAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, head_dim, window_size, use_checkpoint=False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_groups = num_heads // num_kv_heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.q_proj = nn.Linear(d_model, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, d_model, bias=False)

    def _expand_kv(self, k, v, B):
        if self.num_groups == 1:
            return k, v
        k_e = k.unsqueeze(2).expand(-1, -1, self.num_groups, -1, -1).reshape(
            B, self.num_heads, -1, self.head_dim)
        v_e = v.unsqueeze(2).expand(-1, -1, self.num_groups, -1, -1).reshape(
            B, self.num_heads, -1, self.head_dim)
        return k_e, v_e

    def _compute_importance_chunked(self, q, k_e, attn_bool, q_mask, k_mask):
        B, H, T_q, _ = q.shape
        T_kv = k_e.shape[2]
        if T_q == 0 or T_kv == 0:
            return torch.zeros(B, T_kv, device=q.device, dtype=torch.float32)
        C = torch.zeros(B, T_kv, device=q.device, dtype=torch.float32)
        q_chunk = 128
        k_t = k_e.transpose(-2, -1)
        for qs in range(0, T_q, q_chunk):
            qe = min(qs + q_chunk, T_q)
            scores_c = torch.matmul(q[:, :, qs:qe, :], k_t) * self.scale
            mask_c = attn_bool[:, :, qs:qe, :]
            has_valid_c = mask_c.any(dim=-1, keepdim=True)
            safe_mask_c = mask_c.clone()
            safe_mask_c[..., 0] = safe_mask_c[..., 0] | (~has_valid_c.squeeze(-1))
            scores_c = scores_c.masked_fill(~safe_mask_c, float('-inf'))
            attn_w_c = F.softmax(scores_c.float(), dim=-1)
            attn_w_c = torch.where(has_valid_c, attn_w_c, torch.zeros_like(attn_w_c))
            if q_mask is not None:
                w_sum_c = (attn_w_c * q_mask[:, None, qs:qe, None]).sum(dim=(1, 2))
            else:
                w_sum_c = attn_w_c.sum(dim=(1, 2))
            C = C + w_sum_c
        if k_mask is not None:
            C = C * k_mask.float()
        return C

    def _compute_causal_routing_scores(self, q, k_e, attn_bool, q_pos_t, k_pos_t,
                                       q_mask, k_mask, seq_idx, routing_spec):
        block_size, num_blocks, kv_pos_spec, seq_spec = routing_spec
        B, H, T_q, _ = q.shape
        T_kv = k_e.shape[2]
        device = q.device
        max_cap = num_blocks * block_size
        if T_q == 0 or T_kv == 0 or num_blocks == 0:
            empty_f = torch.full((B, T_q, num_blocks), float('-inf'), device=device, dtype=torch.float32)
            empty_b = torch.zeros(B, T_q, num_blocks, device=device, dtype=torch.bool)
            c_last = torch.zeros(B, T_kv, device=device, dtype=torch.float32)
            return empty_f, empty_f.clone(), empty_b, c_last

        q_p = kv_pos_spec[:, :T_q] if kv_pos_spec is not None else (
            q_pos_t if q_pos_t is not None else torch.arange(T_q, device=device).unsqueeze(0).expand(B, T_q)
        )
        k_p = kv_pos_spec[:, :T_kv] if kv_pos_spec is not None else (
            k_pos_t if k_pos_t is not None else torch.arange(T_kv, device=device).unsqueeze(0).expand(B, T_kv)
        )
        q_valid = (q_p >= 0) & (q_p < max_cap)
        if q_mask is not None:
            q_valid = q_valid & (q_mask > 0)
        k_valid = (k_p >= 0) & (k_p < max_cap)
        if k_mask is not None:
            k_valid = k_valid & k_mask.bool()
        seq_t = seq_spec if seq_spec is not None else seq_idx
        if seq_t is not None:
            q_seq = seq_t[:, :T_q].to(torch.int32)
            k_seq = seq_t[:, :T_kv].to(torch.int32)
            q_valid = q_valid & (q_seq >= 0)
            k_valid = k_valid & (k_seq >= 0)
        else:
            q_seq = None
            k_seq = None

        pos_safe_k = torch.where(k_valid, k_p.clamp(min=0, max=max(max_cap - 1, 0)), torch.full_like(k_p, max_cap))
        C_running = torch.zeros(B, T_kv, device=device, dtype=torch.float32)
        k_t = k_e.transpose(-2, -1)
        q_chunk = 128
        sa_list = []
        sb_list = []
        vb_list = []
        for qs in range(0, T_q, q_chunk):
            qe = min(qs + q_chunk, T_q)
            Qc = qe - qs
            scores_c = torch.matmul(q[:, :, qs:qe, :], k_t) * self.scale
            mask_c = attn_bool[:, :, qs:qe, :]
            has_valid_c = mask_c.any(dim=-1, keepdim=True)
            safe_mask_c = mask_c.clone()
            safe_mask_c[..., 0] = safe_mask_c[..., 0] | (~has_valid_c.squeeze(-1))
            scores_c = scores_c.masked_fill(~safe_mask_c, float('-inf'))
            attn_w_c = F.softmax(scores_c.float(), dim=-1)
            attn_w_c = torch.where(has_valid_c, attn_w_c, torch.zeros_like(attn_w_c))
            q_m_c = q_valid[:, qs:qe].float()
            w_sum_c = attn_w_c.sum(dim=1) * q_m_c[:, :, None]
            C_c = C_running[:, None, :] + w_sum_c.cumsum(dim=1)
            C_running = C_c[:, -1, :]

            vis_c = (k_p[:, None, :] <= q_p[:, qs:qe, None]) & k_valid[:, None, :] & q_valid[:, qs:qe, None]
            if q_seq is not None:
                vis_c = vis_c & (q_seq[:, qs:qe, None] == k_seq[:, None, :])

            pos_exp = pos_safe_k[:, None, :].expand(-1, Qc, -1)
            C_comp_c = torch.zeros(B, Qc, max_cap + 1, device=device, dtype=torch.float32).scatter_(
                2, pos_exp, C_c * vis_c.float())[:, :, :max_cap]
            V_comp_c = torch.zeros(B, Qc, max_cap + 1, device=device, dtype=torch.bool).scatter_(
                2, pos_exp, vis_c)[:, :, :max_cap]

            C_blk = C_comp_c.view(B, Qc, num_blocks, block_size)
            V_blk = V_comp_c.view(B, Qc, num_blocks, block_size)
            cnt_blk = V_blk.sum(dim=-1)
            vb_c = cnt_blk > 0
            sa_c = C_blk.masked_fill(~V_blk, float('-inf')).max(dim=-1).values
            sb_c = (C_blk.sum(dim=-1) / cnt_blk.clamp(min=1).float()).masked_fill(~vb_c, float('-inf'))
            sa_list.append(sa_c)
            sb_list.append(sb_c)
            vb_list.append(vb_c)

        score_a = torch.cat(sa_list, dim=1)
        score_b = torch.cat(sb_list, dim=1)
        valid_blk = torch.cat(vb_list, dim=1)
        C_last = C_running * k_valid.float()
        return score_a, score_b, valid_blk, C_last

    def _core(self, x, past_kv, seq_idx, attention_mask, return_importance,
              past_seq_idx=None, q_pos=None, past_kv_pos=None, routing_spec=None):
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        past_len = 0
        if past_kv is not None:
            past_k, past_v = past_kv
            past_len = past_k.shape[2]
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_kv = (k, v)
        T_q, T_kv = q.shape[2], k.shape[2]

        fast_causal = (
            not return_importance
            and past_kv is None
            and seq_idx is None
            and attention_mask is None
            and q_pos is None
            and past_kv_pos is None
            and T_q > 0
            and (self.window_size is None or self.window_size >= T_q - 1)
        )
        if fast_causal:
            if _HAS_SDPA_GQA and self.num_groups > 1:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            else:
                k_e, v_e = self._expand_kv(k, v, B)
                out = F.scaled_dot_product_attention(q, k_e, v_e, is_causal=True)
            out = out.transpose(1, 2).contiguous().view(B, T_q, self.num_heads * self.head_dim)
            return self.out_proj(out), new_kv, None

        k_e, v_e = self._expand_kv(k, v, B)
        if q_pos is not None or past_kv_pos is not None:
            if q_pos is None:
                q_pos_t = torch.arange(past_len, past_len + T_q, device=x.device).unsqueeze(0).expand(B, T_q)
            else:
                q_pos_t = q_pos
            if past_kv_pos is not None:
                k_pos_t = torch.cat([past_kv_pos, q_pos_t], dim=1)
            elif past_len == 0:
                k_pos_t = q_pos_t
            else:
                past_pos = torch.arange(past_len, device=x.device).unsqueeze(0).expand(B, past_len)
                k_pos_t = torch.cat([past_pos, q_pos_t], dim=1)
            causal = (k_pos_t.unsqueeze(1) >= 0) & (q_pos_t.unsqueeze(2) >= 0) & (k_pos_t.unsqueeze(1) <= q_pos_t.unsqueeze(2))
            if self.window_size is not None:
                causal = causal & ((q_pos_t.unsqueeze(2) - k_pos_t.unsqueeze(1)) <= self.window_size)
            attn_bool = causal.unsqueeze(1)
        else:
            q_pos_t = None
            k_pos_t = None
            causal = _causal_mask(T_q, T_kv, past_len, x.device, self.window_size)
            attn_bool = causal.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
        sb = _seq_boundary_mask(seq_idx, T_q, T_kv, past_len, x.device, past_seq_idx=past_seq_idx)
        if sb is not None:
            attn_bool = attn_bool & sb
        if attention_mask is not None:
            if attention_mask.size(1) >= T_kv and (past_len == 0 or attention_mask.size(1) == T_kv):
                attn_bool = attn_bool & attention_mask[:, None, None, :T_kv].bool()
            elif past_len > 0 and attention_mask.size(1) > T_kv:
                attn_bool = attn_bool & attention_mask[:, None, None, :T_kv].bool()
            elif past_len > 0 and attention_mask.size(1) == T_q:
                past_ok = torch.ones(B, past_len, dtype=torch.bool, device=x.device)
                if past_kv_pos is not None:
                    past_ok = past_kv_pos >= 0
                full_am = torch.cat([past_ok, attention_mask[:, :T_q].bool()], dim=1)
                attn_bool = attn_bool & full_am[:, None, None, :]

        if past_kv is None and seq_idx is None and attention_mask is None and q_pos is None and past_kv_pos is None and T_q > 0 and (self.window_size is None or self.window_size >= T_q - 1):
            if _HAS_SDPA_GQA and self.num_groups > 1:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            else:
                out = F.scaled_dot_product_attention(q, k_e, v_e, is_causal=True)
        else:
            out = _safe_sdpa(q, k_e, v_e, attn_bool, mutate_mask=not return_importance)

        if return_importance:
            with torch.no_grad():
                if attention_mask is not None and attention_mask.size(1) >= past_len + T_q:
                    q_mask = attention_mask[:, past_len:past_len + T_q].float()
                    if q_pos_t is not None:
                        q_mask = q_mask * (q_pos_t >= 0).float()
                elif attention_mask is not None and past_len > 0 and attention_mask.size(1) == T_q:
                    q_mask = attention_mask[:, :T_q].float()
                    if q_pos_t is not None:
                        q_mask = q_mask * (q_pos_t >= 0).float()
                elif q_pos_t is not None:
                    q_mask = (q_pos_t >= 0).float()
                else:
                    q_mask = None

                if attention_mask is not None and attention_mask.size(1) >= T_kv:
                    k_mask = attention_mask[:, :T_kv].bool()
                elif past_len > 0 and attention_mask is not None and attention_mask.size(1) == T_q:
                    past_ok = (past_kv_pos >= 0) if past_kv_pos is not None else torch.ones(B, past_len, dtype=torch.bool, device=x.device)
                    k_mask = torch.cat([past_ok, attention_mask[:, :T_q].bool()], dim=1)
                elif k_pos_t is not None:
                    k_mask = k_pos_t >= 0
                else:
                    k_mask = None

                if routing_spec is not None:
                    C = self._compute_causal_routing_scores(
                        q, k_e, attn_bool, q_pos_t, k_pos_t, q_mask, k_mask, seq_idx, routing_spec)
                else:
                    C = self._compute_importance_chunked(q, k_e, attn_bool, q_mask, k_mask)
        else:
            C = None

        out = out.transpose(1, 2).contiguous().view(B, T_q, self.num_heads * self.head_dim)
        return self.out_proj(out), new_kv, C

    def forward(self, x, past_kv=None, seq_idx=None, attention_mask=None,
                return_importance=False, past_seq_idx=None, q_pos=None, past_kv_pos=None,
                routing_spec=None):
        if self.use_checkpoint and self.training and torch.is_grad_enabled() and past_kv is None:
            if return_importance and routing_spec is not None:
                def _fn(inp):
                    out, _, c_tup = self._core(
                        inp, None, seq_idx, attention_mask, True,
                        None, q_pos, None, routing_spec=routing_spec)
                    return out, c_tup[0], c_tup[1], c_tup[2], c_tup[3]
                out, sa, sb, vb, c_last = checkpoint(_fn, x, use_reentrant=True)
                return out, None, (sa, sb, vb, c_last)
            def _fn(inp):
                return self._core(inp, None, seq_idx, attention_mask, return_importance,
                                  None, q_pos, None, routing_spec=routing_spec)
            return checkpoint(_fn, x, use_reentrant=True)
        return self._core(x, past_kv, seq_idx, attention_mask, return_importance,
                          past_seq_idx=past_seq_idx, q_pos=q_pos, past_kv_pos=past_kv_pos,
                          routing_spec=routing_spec)


class ThoughtGQAAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, head_dim,
                 block_size, top_k_heads, max_blocks, max_full, use_checkpoint=False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_groups = num_heads // num_kv_heads
        self.scale = head_dim ** -0.5
        self.block_size = block_size
        self.top_k_heads = top_k_heads
        self.max_blocks = max_blocks
        self.max_full = max_full
        self.use_checkpoint = use_checkpoint
        self.q_proj = nn.Linear(d_model, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, d_model, bias=False)

    def _build_head_mask(self, q_mean, k_blocks_mean, selected, advanced):
        B, H, hd = q_mean.shape
        NB = k_blocks_mean.shape[1]
        kv_h = k_blocks_mean.shape[2]
        ng = H // kv_h
        k_h = k_blocks_mean.unsqueeze(3).expand(-1, -1, -1, ng, -1).reshape(B, NB, H, hd)
        scores = torch.einsum('bhd,bnhd->bhn', q_mean.float(), k_h.float())
        K = min(self.top_k_heads, H)
        topk_idx = scores.topk(K, dim=1).indices
        head_ok = torch.zeros(B, H, NB, dtype=torch.bool, device=q_mean.device)
        head_ok.scatter_(1, topk_idx, torch.ones_like(topk_idx, dtype=torch.bool))
        head_ok = torch.where(advanced.unsqueeze(1), torch.ones_like(head_ok), head_ok)
        head_ok = head_ok & selected.unsqueeze(1)
        return head_ok

    def _expand_kv(self, k, v, B):
        if self.num_groups == 1:
            return k, v
        k_e = k.unsqueeze(2).expand(-1, -1, self.num_groups, -1, -1).reshape(
            B, self.num_heads, -1, self.head_dim)
        v_e = v.unsqueeze(2).expand(-1, -1, self.num_groups, -1, -1).reshape(
            B, self.num_heads, -1, self.head_dim)
        return k_e, v_e

    def _segmented_prefix_mean_4d(self, x_f, valid_tok, is_seg_start):
        B, C, T, hd = x_f.shape
        idx_t = torch.arange(T, device=x_f.device).unsqueeze(0).expand(B, T)
        start_idx = torch.where(is_seg_start, idx_t, torch.zeros_like(idx_t)).cummax(dim=1).values
        x_cs = x_f.cumsum(dim=2)
        x_prev = F.pad(x_cs[:, :, :-1, :], (0, 0, 1, 0), value=0.0)
        cnt_cs = valid_tok.float().cumsum(dim=1)
        cnt_prev = F.pad(cnt_cs[:, :-1], (1, 0), value=0.0)
        base_x = x_prev.gather(2, start_idx[:, None, :, None].expand(-1, C, -1, hd))
        base_cnt = cnt_prev.gather(1, start_idx)
        return (x_cs - base_x) / (cnt_cs - base_cnt).clamp(min=1.0)[:, None, :, None]

    def _core_prefill(self, x, seq_idx, attention_mask, selected, advanced, build_cache,
                      real_pos=None, selected_per_seq=None, advanced_per_seq=None):
        B, T, D = x.shape
        device = x.device
        H = self.num_heads
        kv_h = self.num_kv_heads
        hd = self.head_dim
        bsz = self.block_size

        q = self.q_proj(x).view(B, T, H, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, kv_h, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, kv_h, hd).transpose(1, 2)

        if selected is None:
            NB = max((T + bsz - 1) // bsz, 1)
        else:
            NB = selected.shape[-1]

        pos_arr = real_pos if real_pos is not None else torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        valid_tok = (pos_arr >= 0) & (pos_arr < NB * bsz)
        if attention_mask is not None:
            valid_tok = valid_tok & attention_mask[:, :T].bool()
        if seq_idx is not None:
            seq_t = seq_idx[:, :T].to(torch.int32)
            valid_tok = valid_tok & (seq_t >= 0)
        else:
            seq_t = None

        if T == 0:
            out = torch.zeros(B, 0, D, device=device, dtype=x.dtype)
            new_cache = None
            if build_cache:
                sel_c = torch.zeros(B, NB, dtype=torch.bool, device=device) if selected is None else (
                    selected[:, -1] if selected.dim() == 3 else selected)
                adv_c = torch.zeros(B, NB, dtype=torch.bool, device=device) if advanced is None else (
                    advanced[:, -1] if advanced.dim() == 3 else advanced)
                new_cache = self._build_cache_from_full(
                    k, v, q, T, sel_c, adv_c, attention_mask, device,
                    seq_idx=seq_idx, real_pos=real_pos)
            return out, new_cache

        pos_safe = pos_arr.clamp(min=0, max=max(NB * bsz - 1, 0))
        tok_blk = pos_safe // bsz
        v_f = valid_tok.float()
        idx_t = torch.arange(T, device=device).unsqueeze(0).expand(B, T)

        if selected is None:
            blk_rng = torch.arange(NB, device=device).view(1, 1, NB)
            sel_3d = (blk_rng <= tok_blk.unsqueeze(-1)) & valid_tok.unsqueeze(-1)
            adv_3d = torch.zeros_like(sel_3d)
        elif selected.dim() == 2:
            if selected_per_seq is not None and seq_t is not None:
                num_seqs = selected_per_seq.shape[1]
                s_idx_safe = (seq_t - 1).clamp(min=0, max=max(num_seqs - 1, 0))
                sel_3d = selected_per_seq.gather(1, s_idx_safe.unsqueeze(-1).expand(-1, -1, NB)) & valid_tok.unsqueeze(-1)
                adv_3d = advanced_per_seq.gather(1, s_idx_safe.unsqueeze(-1).expand(-1, -1, NB)) & valid_tok.unsqueeze(-1)
            else:
                sel_3d = selected.unsqueeze(1).expand(-1, T, -1)
                adv_3d = advanced.unsqueeze(1).expand(-1, T, -1)
        else:
            sel_3d = selected
            adv_3d = advanced

        sel_eff = sel_3d.clone()
        sel_eff.scatter_(2, tok_blk.unsqueeze(-1), sel_eff.gather(2, tok_blk.unsqueeze(-1)) | valid_tok.unsqueeze(-1))

        if seq_t is not None:
            seq_eff = torch.where(valid_tok, seq_t, torch.full_like(seq_t, -1))
            last_v_idx = torch.where(valid_tok, idx_t, torch.zeros_like(idx_t)).cummax(dim=1).values
            ff_seq = seq_eff.gather(1, last_v_idx)
            prev_ff = F.pad(ff_seq[:, :-1], (1, 0), value=-999999)
            is_seq_start = valid_tok & (ff_seq != prev_ff)
            q_prefix_mean = self._segmented_prefix_mean_4d(
                q.detach().float() * v_f[:, None, :, None], valid_tok, is_seq_start)
        else:
            seq_eff = None
            is_seq_start = torch.zeros(B, T, dtype=torch.bool, device=device)
            is_seq_start[:, 0] = valid_tok[:, 0]
            q_cs = (q.detach().float() * v_f[:, None, :, None]).cumsum(dim=2)
            cnt_cs = v_f.cumsum(dim=1).clamp(min=1.0)
            q_prefix_mean = q_cs / cnt_cs[:, None, :, None]

        blk_eff = torch.where(valid_tok, tok_blk, torch.full_like(tok_blk, -1))
        last_vb_idx = torch.where(valid_tok, idx_t, torch.zeros_like(idx_t)).cummax(dim=1).values
        ff_blk = blk_eff.gather(1, last_vb_idx)
        prev_ff_blk = F.pad(ff_blk[:, :-1], (1, 0), value=-999999)
        is_blk_start = valid_tok & (ff_blk != prev_ff_blk)
        is_seg_start_k = is_seq_start | is_blk_start

        k_f_seq = k.detach().float() * v_f[:, None, :, None]
        k_bm_self = self._segmented_prefix_mean_4d(k_f_seq, valid_tok, is_seg_start_k)
        if self.num_groups > 1:
            k_self_h = k_bm_self.unsqueeze(2).expand(-1, -1, self.num_groups, -1, -1).reshape(B, H, T, hd)
        else:
            k_self_h = k_bm_self

        blk_idx_exp = tok_blk.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, kv_h, hd)
        k_t_f = k.detach().transpose(1, 2).float()

        if seq_eff is not None:
            uniq_seqs = torch.unique(seq_eff[valid_tok]) if valid_tok.any() else seq_eff.new_empty(0)
            seq_list = uniq_seqs.tolist()
        else:
            seq_list = []

        if len(seq_list) <= 1:
            k_sum_buf = torch.zeros(B, NB, kv_h, hd, device=device, dtype=torch.float32).scatter_add(
                1, blk_idx_exp, k_t_f * v_f[:, :, None, None])
            cnt_buf = torch.zeros(B, NB, device=device, dtype=torch.float32).scatter_add(1, tok_blk, v_f)
            valid_blk_global = cnt_buf > 0
            k_bm_past = k_sum_buf / cnt_buf.clamp(min=1.0)[:, :, None, None]
            if self.num_groups > 1:
                k_bm_h = k_bm_past.unsqueeze(3).expand(-1, -1, -1, self.num_groups, -1).reshape(B, NB, H, hd)
            else:
                k_bm_h = k_bm_past
            seq_k_bm_list = None
        else:
            seq_k_bm_list = []
            for s_val in seq_list:
                m_s = valid_tok & (seq_eff == s_val)
                v_fs = m_s.float()
                k_sum_s = torch.zeros(B, NB, kv_h, hd, device=device, dtype=torch.float32).scatter_add(
                    1, blk_idx_exp, k_t_f * v_fs[:, :, None, None])
                cnt_s = torch.zeros(B, NB, device=device, dtype=torch.float32).scatter_add(1, tok_blk, v_fs)
                vb_s = cnt_s > 0
                k_bm_s = k_sum_s / cnt_s.clamp(min=1.0)[:, :, None, None]
                if self.num_groups > 1:
                    k_bm_sh = k_bm_s.unsqueeze(3).expand(-1, -1, -1, self.num_groups, -1).reshape(B, NB, H, hd)
                else:
                    k_bm_sh = k_bm_s
                seq_k_bm_list.append((s_val, k_bm_sh, vb_s))

        k_e, v_e = self._expand_kv(k, v, B)
        blk_range = torch.arange(NB, device=device).view(1, 1, NB)
        tok_blk_k_exp = tok_blk.view(B, 1, 1, T)
        K_top = min(self.top_k_heads, H)

        q_chunk = 256
        out_chunks = []
        for qs in range(0, T, q_chunk):
            qe = min(qs + q_chunk, T)
            Qc = qe - qs
            q_pm_c = q_prefix_mean[:, :, qs:qe, :]

            if seq_k_bm_list is None:
                scores_c = torch.einsum('bhqd,bnhd->bhqn', q_pm_c, k_bm_h).float()
                blk_valid_c = valid_blk_global.unsqueeze(1).expand(-1, Qc, -1)
            else:
                scores_c = torch.zeros(B, H, Qc, NB, device=device, dtype=torch.float32)
                blk_valid_c = torch.zeros(B, Qc, NB, device=device, dtype=torch.bool)
                for s_val, k_bm_sh, vb_s in seq_k_bm_list:
                    qm_s = (seq_eff[:, qs:qe] == s_val) & valid_tok[:, qs:qe]
                    if qm_s.any():
                        sc_s = torch.einsum('bhqd,bnhd->bhqn', q_pm_c, k_bm_sh).float()
                        scores_c = torch.where(qm_s[:, None, :, None], sc_s, scores_c)
                        blk_valid_c = torch.where(qm_s[:, :, None], vb_s.unsqueeze(1), blk_valid_c)

            diag_score = (q_pm_c * k_self_h[:, :, qs:qe, :]).sum(dim=-1)
            diag_idx = tok_blk[:, qs:qe].unsqueeze(1).unsqueeze(-1).expand(-1, H, -1, -1)
            scores_c = scores_c.scatter(3, diag_idx, diag_score.unsqueeze(-1).to(scores_c.dtype))

            causal_blk_c = (blk_range <= tok_blk[:, qs:qe].unsqueeze(-1)) & valid_tok[:, qs:qe].unsqueeze(-1) & blk_valid_c
            sel_c = sel_eff[:, qs:qe, :] & causal_blk_c
            adv_c = adv_3d[:, qs:qe, :] & causal_blk_c

            topk_idx_c = scores_c.topk(K_top, dim=1).indices
            head_ok_c = torch.zeros(B, H, Qc, NB, dtype=torch.bool, device=device)
            head_ok_c.scatter_(1, topk_idx_c, True)
            head_ok_c = (head_ok_c | adv_c.unsqueeze(1)) & sel_c.unsqueeze(1)

            head_ok_tok_c = head_ok_c.gather(3, tok_blk_k_exp.expand(-1, H, Qc, -1))
            if real_pos is not None:
                causal_c = (pos_arr[:, None, None, :] >= 0) & (pos_arr[:, None, qs:qe, None] >= 0) & (
                    pos_arr[:, None, None, :] <= pos_arr[:, None, qs:qe, None])
            else:
                causal_c = _causal_mask(Qc, T, qs, device, None).unsqueeze(0).unsqueeze(0)

            mask_c = head_ok_tok_c & causal_c & valid_tok[:, None, None, :] & valid_tok[:, None, qs:qe, None]
            if seq_eff is not None:
                sb_c = (seq_eff[:, qs:qe, None] == seq_eff[:, None, :]) & (seq_eff[:, qs:qe, None] >= 0) & (seq_eff[:, None, :] >= 0)
                mask_c = mask_c & sb_c.unsqueeze(1)

            out_c = _safe_sdpa(q[:, :, qs:qe, :], k_e, v_e, mask_c, mutate_mask=True)
            out_chunks.append(out_c)

        out = torch.cat(out_chunks, dim=2)
        out = out.transpose(1, 2).contiguous().view(B, T, H * hd)
        out = self.out_proj(out)

        new_cache = None
        if build_cache:
            if sel_eff.dim() == 3:
                has_v = valid_tok.any(dim=1)
                last_v_pos = torch.where(valid_tok, idx_t, torch.zeros_like(idx_t)).max(dim=1).values
                b_all = torch.arange(B, device=device)
                sel_last = torch.where(has_v.unsqueeze(1), sel_eff[b_all, last_v_pos], torch.zeros(B, NB, dtype=torch.bool, device=device))
                adv_last = torch.where(has_v.unsqueeze(1), adv_3d[b_all, last_v_pos], torch.zeros(B, NB, dtype=torch.bool, device=device))
            else:
                sel_last = sel_eff
                adv_last = adv_3d
            new_cache = self._build_cache_from_full(
                k, v, q, T, sel_last, adv_last, attention_mask, device,
                seq_idx=seq_idx, real_pos=real_pos)
        return out, new_cache

    def forward_prefill(self, x, seq_idx=None, attention_mask=None,
                        selected=None, advanced=None, build_cache=False,
                        real_pos=None, selected_per_seq=None, advanced_per_seq=None):
        if self.use_checkpoint and self.training and torch.is_grad_enabled() and not build_cache:
            def _fn(inp):
                return self._core_prefill(
                    inp, seq_idx, attention_mask, selected, advanced, False,
                    real_pos=real_pos, selected_per_seq=selected_per_seq,
                    advanced_per_seq=advanced_per_seq)[0]
            return checkpoint(_fn, x, use_reentrant=True), None
        return self._core_prefill(
            x, seq_idx, attention_mask, selected, advanced, build_cache,
            real_pos=real_pos, selected_per_seq=selected_per_seq,
            advanced_per_seq=advanced_per_seq)

    def _build_cache_from_full(self, k, v, q, T, selected, advanced, attention_mask, device,
                               seq_idx=None, real_pos=None):
        B, kv_heads, _, hd = k.shape
        H = q.shape[1]
        NB = max(selected.shape[1], self.max_blocks)
        selected_p = torch.zeros(B, NB, dtype=torch.bool, device=device)
        advanced_p = torch.zeros(B, NB, dtype=torch.bool, device=device)
        S = selected.shape[1]
        selected_p[:, :S] = selected
        advanced_p[:, :S] = advanced

        total_cap = NB * self.block_size
        if total_cap < T:
            raise ValueError(f"T={T} exceeds NB*block_size={total_cap}")

        if real_pos is not None:
            valid_tok = (real_pos >= 0) & (real_pos < total_cap)
            if attention_mask is not None:
                valid_tok = valid_tok & attention_mask[:, :T].bool()
            k_bank_flat = torch.zeros(B, total_cap, kv_heads, hd, device=device, dtype=k.dtype)
            v_bank_flat = torch.zeros(B, total_cap, kv_heads, hd, device=device, dtype=v.dtype)
            blk_mask_flat = torch.zeros(B, total_cap, device=device, dtype=torch.bool)
            if valid_tok.any():
                b_idx, t_idx = valid_tok.nonzero(as_tuple=True)
                pos_idx = real_pos[b_idx, t_idx]
                k_bank_flat[b_idx, pos_idx] = k.transpose(1, 2)[b_idx, t_idx]
                v_bank_flat[b_idx, pos_idx] = v.transpose(1, 2)[b_idx, t_idx]
                blk_mask_flat[b_idx, pos_idx] = True
            k_bank = k_bank_flat.view(B, NB, self.block_size, kv_heads, hd).contiguous()
            v_bank = v_bank_flat.view(B, NB, self.block_size, kv_heads, hd).contiguous()
            am_blk_bool = blk_mask_flat.view(B, NB, self.block_size).contiguous()
            row_seq_lens = valid_tok.long().sum(dim=1)
        else:
            pad_len = total_cap - T
            k_p = F.pad(k, (0, 0, 0, pad_len)) if pad_len > 0 else k
            v_p = F.pad(v, (0, 0, 0, pad_len)) if pad_len > 0 else v
            k_bank = k_p.view(B, kv_heads, NB, self.block_size, hd).permute(0, 2, 3, 1, 4).contiguous()
            v_bank = v_p.view(B, kv_heads, NB, self.block_size, hd).permute(0, 2, 3, 1, 4).contiguous()
            if attention_mask is not None:
                am = attention_mask[:, :T].float()
            else:
                am = torch.ones(B, T, device=device, dtype=torch.float32)
            if pad_len > 0:
                am = F.pad(am, (0, pad_len), value=0.0)
            am_blk_bool = am.view(B, NB, self.block_size).bool()
            m5 = am_blk_bool.unsqueeze(-1).unsqueeze(-1)
            k_bank = torch.where(m5, k_bank, torch.zeros_like(k_bank))
            v_bank = torch.where(m5, v_bank, torch.zeros_like(v_bank))
            row_seq_lens = am_blk_bool.view(B, -1).long().sum(dim=1)
            valid_tok = am_blk_bool.view(B, -1)[:, :T] if T > 0 else torch.zeros(B, 0, dtype=torch.bool, device=device)

        actual_counts = am_blk_bool.long().sum(-1)

        blk_seq_ids = None
        full_seq_ids = None
        active_seq_ids = None
        if seq_idx is not None:
            seq_t = seq_idx[:, :T].to(torch.int32)
            if attention_mask is not None:
                seq_t = torch.where(attention_mask[:, :T].bool(), seq_t, torch.full_like(seq_t, -1))
            if real_pos is not None:
                blk_seq_flat = torch.full((B, total_cap), -1, device=device, dtype=torch.int32)
                valid_s = (real_pos >= 0) & (real_pos < total_cap) & (seq_t >= 0)
                if valid_s.any():
                    b_idx, t_idx = valid_s.nonzero(as_tuple=True)
                    blk_seq_flat[b_idx, real_pos[b_idx, t_idx]] = seq_t[b_idx, t_idx]
                blk_seq_ids = blk_seq_flat.view(B, NB, self.block_size)
            else:
                pad_len = total_cap - T
                seq_p = F.pad(seq_t, (0, pad_len), value=-1) if pad_len > 0 else seq_t
                blk_seq_ids = seq_p.view(B, NB, self.block_size)
            pos_grid = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
            valid_seq = seq_t >= 0
            last_valid_pos = torch.where(valid_seq, pos_grid, torch.full_like(pos_grid, -1)).max(dim=1).values.clamp(min=0)
            active_seq_ids = seq_t[torch.arange(B, device=device), last_valid_pos]
            active_seq_ids = torch.where(valid_seq.any(dim=1), active_seq_ids, torch.full_like(active_seq_ids, -1))
            sum_mask_b = am_blk_bool & (blk_seq_ids == active_seq_ids.view(B, 1, 1)) & (active_seq_ids.view(B, 1, 1) >= 0)
            full_seq_ids = torch.full((B, self.max_full, self.block_size), -1, device=device, dtype=torch.int32)
            q_act_mask = valid_tok & (seq_t == active_seq_ids.view(B, 1)) & (active_seq_ids.view(B, 1) >= 0)
        else:
            sum_mask_b = am_blk_bool
            q_act_mask = valid_tok

        q_sum = (q.float() * q_act_mask[:, None, :, None].float()).sum(dim=2)
        q_cnt = q_act_mask.float().sum(dim=1)

        slot_pos = torch.arange(total_cap, device=device).view(1, NB, self.block_size)
        sum_max_pos = torch.where(sum_mask_b, slot_pos, torch.full_like(slot_pos, -1)).max(dim=-1).values

        sum_counts = sum_mask_b.long().sum(-1)
        safe_sum_counts = sum_counts.clamp(min=1).float()
        sm5 = sum_mask_b.unsqueeze(-1).unsqueeze(-1).float()

        k_sum_fp32 = (k_bank.float() * sm5).sum(2) / safe_sum_counts.unsqueeze(-1).unsqueeze(-1)
        v_sum_fp32 = (v_bank.float() * sm5).sum(2) / safe_sum_counts.unsqueeze(-1).unsqueeze(-1)

        cache = {
            "k_bank": k_bank,
            "v_bank": v_bank,
            "blk_mask": am_blk_bool,
            "blk_seq_ids": blk_seq_ids,
            "active_seq_ids": active_seq_ids,
            "k_sum": k_sum_fp32.to(k.dtype),
            "v_sum": v_sum_fp32.to(v.dtype),
            "k_sum_fp32": k_sum_fp32,
            "v_sum_fp32": v_sum_fp32,
            "blk_lens": actual_counts,
            "sum_lens": sum_counts,
            "sum_max_pos": sum_max_pos,
            "q_sum": q_sum,
            "q_cnt": q_cnt,
            "k_full": torch.zeros(B, self.max_full, self.block_size, kv_heads, hd,
                                  device=device, dtype=k.dtype),
            "v_full": torch.zeros(B, self.max_full, self.block_size, kv_heads, hd,
                                  device=device, dtype=v.dtype),
            "full_mask": torch.zeros(B, self.max_full, self.block_size, device=device, dtype=torch.bool),
            "full_seq_ids": full_seq_ids,
            "full_ids": torch.full((B, self.max_full), -1, device=device, dtype=torch.long),
            "full_lens": torch.zeros(B, self.max_full, device=device, dtype=torch.long),
            "seq_len": T,
            "row_seq_lens": row_seq_lens,
            "selected": selected_p,
            "advanced": advanced_p,
        }

        for b in range(B):
            sel_blocks = selected_p[b].nonzero(as_tuple=True)[0][:self.max_full].tolist()
            for slot, blk in enumerate(sel_blocks):
                n = int(actual_counts[b, blk].item())
                cache["full_ids"][b, slot] = blk
                cache["full_lens"][b, slot] = n
                cache["full_mask"][b, slot] = am_blk_bool[b, blk]
                cache["k_full"][b, slot] = k_bank[b, blk]
                cache["v_full"][b, slot] = v_bank[b, blk]
                if full_seq_ids is not None:
                    full_seq_ids[b, slot] = blk_seq_ids[b, blk]
        return cache

    def _expand_cache_blocks(self, cache, new_nb):
        old_nb = cache["selected"].shape[1]
        if new_nb <= old_nb:
            return
        extra = new_nb - old_nb
        B = cache["selected"].shape[0]
        bsz = self.block_size
        kv_h = self.num_kv_heads
        hd = self.head_dim
        device = cache["selected"].device
        cache["k_bank"] = torch.cat(
            [cache["k_bank"], torch.zeros(B, extra, bsz, kv_h, hd, device=device, dtype=cache["k_bank"].dtype)],
            dim=1,
        )
        cache["v_bank"] = torch.cat(
            [cache["v_bank"], torch.zeros(B, extra, bsz, kv_h, hd, device=device, dtype=cache["v_bank"].dtype)],
            dim=1,
        )
        cache["blk_mask"] = torch.cat(
            [cache["blk_mask"], torch.zeros(B, extra, bsz, device=device, dtype=torch.bool)],
            dim=1,
        )
        if cache["blk_seq_ids"] is not None:
            cache["blk_seq_ids"] = torch.cat(
                [cache["blk_seq_ids"], torch.full((B, extra, bsz), -1, device=device, dtype=torch.int32)],
                dim=1,
            )
        cache["k_sum"] = torch.cat(
            [cache["k_sum"], torch.zeros(B, extra, kv_h, hd, device=device, dtype=cache["k_sum"].dtype)],
            dim=1,
        )
        cache["v_sum"] = torch.cat(
            [cache["v_sum"], torch.zeros(B, extra, kv_h, hd, device=device, dtype=cache["v_sum"].dtype)],
            dim=1,
        )
        cache["k_sum_fp32"] = torch.cat(
            [cache["k_sum_fp32"], torch.zeros(B, extra, kv_h, hd, device=device, dtype=torch.float32)],
            dim=1,
        )
        cache["v_sum_fp32"] = torch.cat(
            [cache["v_sum_fp32"], torch.zeros(B, extra, kv_h, hd, device=device, dtype=torch.float32)],
            dim=1,
        )
        cache["blk_lens"] = torch.cat(
            [cache["blk_lens"], torch.zeros(B, extra, device=device, dtype=torch.long)],
            dim=1,
        )
        cache["sum_lens"] = torch.cat(
            [cache["sum_lens"], torch.zeros(B, extra, device=device, dtype=torch.long)],
            dim=1,
        )
        if "sum_max_pos" in cache:
            cache["sum_max_pos"] = torch.cat(
                [cache["sum_max_pos"], torch.full((B, extra), -1, device=device, dtype=torch.long)],
                dim=1,
            )
        cache["selected"] = torch.cat(
            [cache["selected"], torch.zeros(B, extra, device=device, dtype=torch.bool)],
            dim=1,
        )
        cache["advanced"] = torch.cat(
            [cache["advanced"], torch.zeros(B, extra, device=device, dtype=torch.bool)],
            dim=1,
        )

    def _reeval_slots(self, cache, reeval_mask=None):
        B = cache["full_ids"].shape[0]
        F_slots = cache["full_ids"].shape[1]
        NB = cache["selected"].shape[1]
        device = cache["full_ids"].device
        sel = cache["selected"]

        old_ids = cache["full_ids"]
        old_ids_safe = old_ids.clamp(min=0)
        blk_is_sel = sel.gather(1, old_ids_safe)
        keep = blk_is_sel & (old_ids >= 0)

        new_ids = torch.where(keep, old_ids, torch.full_like(old_ids, -1))
        free = (new_ids == -1)

        has_full = (new_ids.unsqueeze(-1) == torch.arange(NB, device=device).view(1, 1, NB)).any(1)
        need_alloc = sel & ~has_full
        need_rank = torch.cumsum(need_alloc.long(), dim=-1) - 1
        free_rank = torch.cumsum(free.long(), dim=-1) - 1

        need_rank_s = need_rank.clone().float()
        need_rank_s[~need_alloc] = float('inf')
        sorted_blocks = need_rank_s.argsort(dim=-1)
        free_rank_s = free_rank.clone().float()
        free_rank_s[~free] = float('inf')
        sorted_slots = free_rank_s.argsort(dim=-1)

        num_need = need_alloc.sum(-1)
        num_free = free.sum(-1)
        num_assign = torch.minimum(num_need, num_free)

        slot_assignment = new_ids.clone()
        b_idx = torch.arange(B, device=device)
        for r in range(F_slots):
            valid = r < num_assign
            if not valid.any():
                break
            v_b = b_idx[valid]
            v_s = sorted_slots[:, r][valid]
            v_blk = sorted_blocks[:, r][valid]
            slot_assignment[v_b, v_s] = v_blk

        if reeval_mask is not None:
            slot_assignment = torch.where(reeval_mask.unsqueeze(1), slot_assignment, old_ids)

        if reeval_mask is not None:
            newly_assigned = (new_ids == -1) & (slot_assignment >= 0) & reeval_mask.unsqueeze(1)
            still_free = (slot_assignment == -1) & reeval_mask.unsqueeze(1)
        else:
            newly_assigned = (new_ids == -1) & (slot_assignment >= 0)
            still_free = slot_assignment == -1

        if newly_assigned.any():
            nb_idx, ns_idx = newly_assigned.nonzero(as_tuple=True)
            blk_idx = slot_assignment[nb_idx, ns_idx]
            cache["k_full"][nb_idx, ns_idx] = cache["k_bank"][nb_idx, blk_idx]
            cache["v_full"][nb_idx, ns_idx] = cache["v_bank"][nb_idx, blk_idx]
            cache["full_mask"][nb_idx, ns_idx] = cache["blk_mask"][nb_idx, blk_idx]
            if cache["full_seq_ids"] is not None and cache["blk_seq_ids"] is not None:
                cache["full_seq_ids"][nb_idx, ns_idx] = cache["blk_seq_ids"][nb_idx, blk_idx]

        if still_free.any():
            fb_idx, fs_idx = still_free.nonzero(as_tuple=True)
            cache["full_mask"][fb_idx, fs_idx] = False
            if cache["full_seq_ids"] is not None:
                cache["full_seq_ids"][fb_idx, fs_idx] = -1

        cache["full_ids"] = slot_assignment
        cache["full_lens"] = cache["full_mask"].long().sum(-1)

    def _write_token_to_cache(self, cache, blk, offset, k_t, v_t, cur_seq_ids=None, write_mask=None):
        B = k_t.shape[0]
        device = k_t.device
        b_all = torch.arange(B, device=device)
        if isinstance(blk, int):
            blk_b = torch.full((B,), blk, device=device, dtype=torch.long)
        elif blk.dim() == 0:
            blk_b = blk.to(device=device, dtype=torch.long).expand(B)
        else:
            blk_b = blk.to(device=device, dtype=torch.long)
        if isinstance(offset, int):
            off_b = torch.full((B,), offset, device=device, dtype=torch.long)
        elif offset.dim() == 0:
            off_b = offset.to(device=device, dtype=torch.long).expand(B)
        else:
            off_b = offset.to(device=device, dtype=torch.long)

        if write_mask is not None:
            active_w = write_mask.bool()
            if not active_w.any().item():
                return
            b_all = b_all[active_w]
            blk_b = blk_b[active_w]
            off_b = off_b[active_w]
            k_t = k_t[active_w]
            v_t = v_t[active_w]
            if cur_seq_ids is not None:
                cur_seq_ids = cur_seq_ids[active_w]
        B_w = b_all.numel()

        was_valid = cache["blk_mask"][b_all, blk_b, off_b]
        if cur_seq_ids is not None and cache["blk_seq_ids"] is not None:
            was_valid_sum = was_valid & (cache["blk_seq_ids"][b_all, blk_b, off_b] == cur_seq_ids)
            cache["blk_seq_ids"][b_all, blk_b, off_b] = cur_seq_ids
        else:
            was_valid_sum = was_valid
        cache["k_bank"][b_all, blk_b, off_b] = k_t
        cache["v_bank"][b_all, blk_b, off_b] = v_t
        cache["blk_mask"][b_all, blk_b, off_b] = True

        add_blk = (~was_valid).long()
        cache["blk_lens"][b_all, blk_b] = cache["blk_lens"][b_all, blk_b] + add_blk

        old_sum_n = cache["sum_lens"][b_all, blk_b]
        need_add_sum = ~was_valid_sum
        new_sum_n = old_sum_n + need_add_sum.long()
        safe_new_n = new_sum_n.clamp(min=1).float().view(B_w, 1, 1)
        old_n_f = old_sum_n.float().view(B_w, 1, 1)
        new_k_sum_fp32 = torch.where(
            need_add_sum.view(B_w, 1, 1),
            (cache["k_sum_fp32"][b_all, blk_b] * old_n_f + k_t.float()) / safe_new_n,
            cache["k_sum_fp32"][b_all, blk_b],
        )
        new_v_sum_fp32 = torch.where(
            need_add_sum.view(B_w, 1, 1),
            (cache["v_sum_fp32"][b_all, blk_b] * old_n_f + v_t.float()) / safe_new_n,
            cache["v_sum_fp32"][b_all, blk_b],
        )
        cache["k_sum_fp32"][b_all, blk_b] = new_k_sum_fp32
        cache["v_sum_fp32"][b_all, blk_b] = new_v_sum_fp32
        cache["sum_lens"][b_all, blk_b] = new_sum_n
        cache["k_sum"][b_all, blk_b] = new_k_sum_fp32.to(cache["k_sum"].dtype)
        cache["v_sum"][b_all, blk_b] = new_v_sum_fp32.to(cache["v_sum"].dtype)
        if "sum_max_pos" in cache:
            pos_written = blk_b * self.block_size + off_b
            cache["sum_max_pos"][b_all, blk_b] = torch.maximum(cache["sum_max_pos"][b_all, blk_b], pos_written)

        full_ids = cache["full_ids"][b_all]
        match = (full_ids == blk_b.unsqueeze(1))
        has_match = match.any(-1)
        match_slot = match.int().argmax(-1)
        is_free = (full_ids == -1)
        has_free = is_free.any(-1)
        free_slot = is_free.int().argmax(-1)
        ids_for_lru = torch.where(full_ids >= 0, full_ids, torch.full_like(full_ids, 10**9))
        lru_slot = ids_for_lru.argmin(-1)
        slot = torch.where(has_match, match_slot, torch.where(has_free, free_slot, lru_slot))

        needs_init = ~has_match
        if needs_init.any():
            b_sel = b_all[needs_init]
            s_sel = slot[needs_init]
            blk_sel = blk_b[needs_init]
            cache["k_full"][b_sel, s_sel] = cache["k_bank"][b_sel, blk_sel]
            cache["v_full"][b_sel, s_sel] = cache["v_bank"][b_sel, blk_sel]
            cache["full_mask"][b_sel, s_sel] = cache["blk_mask"][b_sel, blk_sel]
            if cache["full_seq_ids"] is not None and cache["blk_seq_ids"] is not None:
                cache["full_seq_ids"][b_sel, s_sel] = cache["blk_seq_ids"][b_sel, blk_sel]
            cache["full_ids"][b_sel, s_sel] = blk_sel

        cache["k_full"][b_all, slot, off_b] = k_t
        cache["v_full"][b_all, slot, off_b] = v_t
        cache["full_mask"][b_all, slot, off_b] = True
        if cache["full_seq_ids"] is not None and cur_seq_ids is not None:
            cache["full_seq_ids"][b_all, slot, off_b] = cur_seq_ids
        cache["full_lens"][b_all, slot] = cache["full_mask"][b_all, slot].long().sum(-1)

    def _gather_decode_kv(self, cache, cur_seq_ids=None):
        B = cache["full_ids"].shape[0]
        device = cache["k_full"].device
        F_slots = cache["full_ids"].shape[1]
        NB = cache["selected"].shape[1]
        bsz = self.block_size
        kv_h = self.num_kv_heads
        hd = self.head_dim

        offsets = torch.arange(bsz, device=device)
        full_pos = cache["full_ids"].unsqueeze(-1) * bsz + offsets
        full_valid = (cache["full_ids"].unsqueeze(-1) >= 0) & cache["full_mask"]
        if cache["full_seq_ids"] is not None and cur_seq_ids is not None:
            full_valid = full_valid & (cache["full_seq_ids"] == cur_seq_ids.view(B, 1, 1)) & (cur_seq_ids.view(B, 1, 1) >= 0)

        blocks_range = torch.arange(NB, device=device)
        has_full = (cache["full_ids"].unsqueeze(-1) == blocks_range.view(1, 1, NB)).any(1)
        need_sum = cache["selected"][:, :NB] & ~has_full & (cache["sum_lens"][:, :NB] > 0)
        default_sum_pos = (blocks_range.unsqueeze(0) * bsz + (bsz - 1)).expand(B, -1)
        if "sum_max_pos" in cache:
            smp = cache["sum_max_pos"][:, :NB]
            sum_pos = torch.where(smp >= 0, smp, default_sum_pos)
        else:
            sum_pos = default_sum_pos

        k_full_flat = cache["k_full"].reshape(B, F_slots * bsz, kv_h, hd)
        v_full_flat = cache["v_full"].reshape(B, F_slots * bsz, kv_h, hd)

        k_all = torch.cat([k_full_flat, cache["k_sum"]], dim=1)
        v_all = torch.cat([v_full_flat, cache["v_sum"]], dim=1)
        pos_all = torch.cat([full_pos.reshape(B, -1), sum_pos], dim=1)
        valid_all = torch.cat([full_valid.reshape(B, -1), need_sum], dim=1)
        return k_all, v_all, pos_all, valid_all

    def _decode_step_1(self, x, cache, selected=None, advanced=None, reeval=False,
                       cur_seq_ids=None, reeval_mask=None, chunk_mask=None):
        B, T_new, D = x.shape
        device = x.device
        past_len = cache["seq_len"]
        row_seq_lens_before = cache.get(
            "row_seq_lens",
            torch.full((B,), past_len, device=device, dtype=torch.long),
        )
        if chunk_mask is not None:
            valid_chunk = chunk_mask[:, :1].bool()
            tok_counts = valid_chunk.long().sum(dim=1)
            pos_mat = torch.where(
                valid_chunk,
                row_seq_lens_before.unsqueeze(1),
                torch.full((B, 1), -1, device=device, dtype=torch.long),
            )
            max_new_pos = int(pos_mat.max().item()) if B > 0 else 0
            end_blk_max = max(max_new_pos, 0) // self.block_size
        else:
            valid_chunk = None
            tok_counts = torch.ones(B, device=device, dtype=torch.long)
            pos_mat = row_seq_lens_before.unsqueeze(1)
            end_blk_max = int(pos_mat.max().item()) // self.block_size if B > 0 else 0

        if selected is not None and selected.dim() == 3:
            selected = selected[:, -1, :]
        if advanced is not None and advanced.dim() == 3:
            advanced = advanced[:, -1, :]

        needed_nb = end_blk_max + 1
        if selected is not None:
            needed_nb = max(needed_nb, selected.shape[1])
        self._expand_cache_blocks(cache, needed_nb)
        NB = cache["selected"].shape[1]

        if cur_seq_ids is None:
            cur_seq_ids = cache.get("active_seq_ids", None)
        else:
            cache["active_seq_ids"] = cur_seq_ids

        b_all = torch.arange(B, device=device)
        if reeval_mask is None:
            reeval_rows = torch.full((B,), bool(reeval), device=device, dtype=torch.bool)
        else:
            reeval_rows = reeval_mask.bool()
        any_reeval = bool(reeval_rows.any().item())

        if any_reeval and selected is not None:
            S = selected.shape[1]
            r_col = reeval_rows.unsqueeze(1)
            cache["selected"][:, :S] = torch.where(r_col, selected, cache["selected"][:, :S])
            cache["advanced"][:, :S] = torch.where(r_col, advanced, cache["advanced"][:, :S])
            if S < NB:
                cache["selected"][reeval_rows, S:] = False
                cache["advanced"][reeval_rows, S:] = False
            if valid_chunk is None:
                cache["selected"][b_all, pos_mat[:, 0] // self.block_size] = True
            else:
                vt = valid_chunk[:, 0]
                if vt.any():
                    cache["selected"][b_all[vt], pos_mat[vt, 0] // self.block_size] = True
            self._reeval_slots(cache, reeval_mask=reeval_rows)
        else:
            if valid_chunk is None:
                cache["selected"][b_all, pos_mat[:, 0] // self.block_size] = True
            else:
                vt = valid_chunk[:, 0]
                if vt.any():
                    cache["selected"][b_all[vt], pos_mat[vt, 0] // self.block_size] = True

        q = self.q_proj(x).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k_new = self.k_proj(x).view(B, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v_new = self.v_proj(x).view(B, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)

        pos_b = pos_mat[:, 0].clamp(min=0)
        blk_b = pos_b // self.block_size
        off_b = pos_b % self.block_size
        wm = valid_chunk[:, 0] if valid_chunk is not None else None
        self._write_token_to_cache(
            cache, blk_b, off_b, k_new[:, :, 0, :], v_new[:, :, 0, :],
            cur_seq_ids=cur_seq_ids, write_mask=wm)

        cache["row_seq_lens"] = row_seq_lens_before + tok_counts
        cache["seq_len"] = past_len + 1

        if "q_sum" in cache and "q_cnt" in cache:
            q_cur_f = q[:, :, 0, :].float()
            if valid_chunk is not None:
                v_1 = valid_chunk[:, 0]
                q_sum = torch.where(v_1[:, None, None], cache["q_sum"] + q_cur_f, cache["q_sum"])
                q_cnt = torch.where(v_1, cache["q_cnt"] + 1.0, cache["q_cnt"])
            else:
                q_sum = cache["q_sum"] + q_cur_f
                q_cnt = cache["q_cnt"] + 1.0
            cache["q_sum"] = q_sum
            cache["q_cnt"] = q_cnt
            q_mean = q_sum / q_cnt.clamp(min=1.0)[:, None, None]
        else:
            q_mean = q[:, :, 0, :].float()

        k_all, v_all, pos_all, valid_all = self._gather_decode_kv(cache, cur_seq_ids=cur_seq_ids)
        k_all = k_all.permute(0, 2, 1, 3).contiguous()
        v_all = v_all.permute(0, 2, 1, 3).contiguous()
        k_e, v_e = self._expand_kv(k_all, v_all, B)

        head_ok = self._build_head_mask(q_mean, cache["k_sum"], cache["selected"], cache["advanced"])

        pos_blk = (pos_all // self.block_size).clamp(0, NB - 1)
        head_ok_pos = head_ok.gather(2, pos_blk.unsqueeze(1).expand(-1, self.num_heads, -1))

        causal = (pos_mat.view(B, 1, 1, 1) >= 0) & (pos_mat.view(B, 1, 1, 1) >= pos_all.view(B, 1, 1, -1))
        mask = causal & valid_all.view(B, 1, 1, -1) & head_ok_pos.view(B, self.num_heads, 1, -1)

        out = _safe_sdpa(q, k_e, v_e, mask, mutate_mask=True)
        out = out.transpose(1, 2).contiguous().view(B, 1, self.num_heads * self.head_dim)
        out = self.out_proj(out)
        if valid_chunk is not None:
            out = out * valid_chunk.unsqueeze(-1).to(out.dtype)
        return out, cache

    def _decode_chunk_impl(self, x, cache, selected=None, advanced=None, reeval=False,
                           cur_seq_ids=None, reeval_mask=None, chunk_mask=None):
        T_new = x.shape[1]
        if T_new == 1:
            return self._decode_step_1(
                x, cache, selected=selected, advanced=advanced, reeval=reeval,
                cur_seq_ids=cur_seq_ids, reeval_mask=reeval_mask, chunk_mask=chunk_mask)
        outs = []
        for t in range(T_new):
            sel_t = selected[:, t:t + 1] if (selected is not None and selected.dim() == 3) else (selected if t == 0 else None)
            adv_t = advanced[:, t:t + 1] if (advanced is not None and advanced.dim() == 3) else (advanced if t == 0 else None)
            cm_t = chunk_mask[:, t:t + 1] if chunk_mask is not None else None
            out_t, cache = self._decode_step_1(
                x[:, t:t + 1, :], cache,
                selected=sel_t,
                advanced=adv_t,
                reeval=reeval if t == 0 else False,
                cur_seq_ids=cur_seq_ids,
                reeval_mask=reeval_mask if t == 0 else None,
                chunk_mask=cm_t)
            outs.append(out_t)
        return torch.cat(outs, dim=1), cache

    def forward_decode(self, x, cache, selected=None, advanced=None, reeval=False,
                       cur_seq_ids=None, reeval_mask=None, chunk_mask=None):
        T_new = x.shape[1]
        if T_new == 1:
            return self._decode_step_1(
                x, cache, selected=selected, advanced=advanced, reeval=reeval,
                cur_seq_ids=cur_seq_ids, reeval_mask=reeval_mask, chunk_mask=chunk_mask)
        outs = []
        for s in range(0, T_new, self.block_size):
            e = min(s + self.block_size, T_new)
            cm_s = chunk_mask[:, s:e] if chunk_mask is not None else None
            sel_s = selected[:, s:e] if (selected is not None and selected.dim() == 3) else (selected if s == 0 else None)
            adv_s = advanced[:, s:e] if (advanced is not None and advanced.dim() == 3) else (advanced if s == 0 else None)
            out_s, cache = self._decode_chunk_impl(
                x[:, s:e, :], cache,
                selected=sel_s,
                advanced=adv_s,
                reeval=reeval if s == 0 else False,
                cur_seq_ids=cur_seq_ids,
                reeval_mask=reeval_mask if s == 0 else None,
                chunk_mask=cm_s)
            outs.append(out_s)
        return torch.cat(outs, dim=1), cache


class Mamba2Layer(nn.Module):
    def __init__(self, d_model, d_state=128, d_conv=4, expand=2, headdim=64,
                 ssm_cache_dtype=None, conv_cache_dtype=None):
        super().__init__()
        if not _HAS_MAMBA2:
            raise ImportError("mamba_ssm required: pip install mamba-ssm")
        self.ssm_cache_dtype = _str_to_dtype(ssm_cache_dtype)
        self.conv_cache_dtype = _str_to_dtype(conv_cache_dtype)
        kwargs = dict(d_model=d_model, d_state=d_state, d_conv=d_conv,
                      expand=expand, headdim=headdim)
        if self.ssm_cache_dtype is not None:
            kwargs["ssm_cache_dtype"] = self.ssm_cache_dtype
        if self.conv_cache_dtype is not None:
            kwargs["conv_cache_dtype"] = self.conv_cache_dtype
        try:
            self.mamba = _Mamba2Native(**kwargs)
        except TypeError:
            self.mamba = _Mamba2Native(d_model=d_model, d_state=d_state,
                                        d_conv=d_conv, expand=expand, headdim=headdim)

    def _can_use_seq_idx(self):
        return _HAS_CAUSAL_CONV1D_FN and getattr(self.mamba, "activation", "silu") in ("silu", "swish")

    def _can_use_varlen(self):
        return _HAS_CAUSAL_CONV1D_VARLEN and self._can_use_seq_idx()

    @staticmethod
    def _sanitize_seq_idx(seq_arr):
        seq_i = seq_arr.to(torch.int32)
        if seq_i.numel() == 0:
            return seq_i
        max_pos = seq_i.max().clamp(min=0) + 1
        return torch.where(seq_i < 0, max_pos, seq_i).contiguous()

    def _ensure_cache(self, inference_params, batch_size, x_dtype):
        layer_idx = getattr(self.mamba, "layer_idx", None)
        if inference_params is None or layer_idx is None:
            return
        if layer_idx in inference_params.key_value_memory_dict:
            return
        conv_w = getattr(getattr(self.mamba, "conv1d", None), "weight", None)
        conv_base_dtype = conv_w.dtype if conv_w is not None else x_dtype
        conv_dtype = self.conv_cache_dtype
        if conv_dtype is None:
            conv_dtype = conv_base_dtype if x_dtype == conv_base_dtype else x_dtype
        ssm_dtype = self.ssm_cache_dtype if self.ssm_cache_dtype is not None else x_dtype
        if hasattr(self.mamba, "allocate_inference_cache") and conv_dtype == ssm_dtype:
            conv_state, ssm_state = self.mamba.allocate_inference_cache(
                batch_size, getattr(inference_params, "max_seqlen", 1024), dtype=conv_dtype
            )
            inference_params.key_value_memory_dict[layer_idx] = (conv_state, ssm_state)
            return
        if all(hasattr(self.mamba, attr) for attr in ("d_conv", "conv1d", "nheads", "headdim", "d_state", "in_proj")):
            conv_state = torch.zeros(
                batch_size, self.mamba.d_conv, self.mamba.conv1d.weight.shape[0],
                device=self.mamba.conv1d.weight.device, dtype=conv_dtype,
            ).transpose(1, 2)
            ssm_state = torch.zeros(
                batch_size, self.mamba.nheads, self.mamba.headdim, self.mamba.d_state,
                device=self.mamba.in_proj.weight.device, dtype=ssm_dtype,
            )
            inference_params.key_value_memory_dict[layer_idx] = (conv_state, ssm_state)

    def _expected_step_conv_dtype(self, x):
        if torch.is_autocast_enabled():
            try:
                return torch.get_autocast_dtype(x.device.type)
            except Exception:
                return x.dtype
        if hasattr(self.mamba, "in_proj"):
            return self.mamba.in_proj.weight.dtype
        return x.dtype

    def _step_single_token(self, x_1, inference_params, valid_mask=None):
        layer_idx = getattr(self.mamba, "layer_idx", None)
        if valid_mask is not None and not valid_mask.any().item():
            return torch.zeros_like(x_1)
        if layer_idx is not None and layer_idx in inference_params.key_value_memory_dict:
            conv_state, ssm_state = inference_params.key_value_memory_dict[layer_idx]
            need_restore = (valid_mask is not None) and (not valid_mask.all().item())
            if need_restore:
                conv_prev = conv_state.clone()
                ssm_prev = ssm_state.clone()
            expected_dt = self._expected_step_conv_dtype(x_1)
            if conv_state.dtype != expected_dt:
                conv_tmp = conv_state.to(expected_dt)
                inference_params.key_value_memory_dict[layer_idx] = (conv_tmp, ssm_state)
                out = self.mamba(x_1, inference_params=inference_params)
                conv_state.copy_(conv_tmp.to(conv_state.dtype))
                inference_params.key_value_memory_dict[layer_idx] = (conv_state, ssm_state)
            else:
                out = self.mamba(x_1, inference_params=inference_params)
            if need_restore:
                inv = ~valid_mask
                conv_state[inv] = conv_prev[inv]
                ssm_state[inv] = ssm_prev[inv]
                out = out * valid_mask.view(x_1.shape[0], 1, 1).to(out.dtype)
            return out
        return self.mamba(x_1, inference_params=inference_params)

    def _mask_conv_state_seq_boundary(self, inference_params, seq_idx, attention_mask, T):
        layer_idx = getattr(self.mamba, "layer_idx", None)
        if inference_params is None or seq_idx is None or layer_idx is None:
            return
        if layer_idx not in inference_params.key_value_memory_dict:
            return
        conv_state, _ = inference_params.key_value_memory_dict[layer_idx]
        B = seq_idx.shape[0]
        state_len = conv_state.shape[-1]
        if state_len <= 0 or T <= 0:
            return
        seq_t = seq_idx[:, :T].to(torch.int32)
        tail_seq = torch.full((B, state_len), -1, device=seq_idx.device, dtype=torch.int32)
        if attention_mask is not None:
            am_bool = attention_mask[:, :T].bool()
            lengths = am_bool.long().sum(dim=1)
            rank = am_bool.long().cumsum(dim=1)
            valid_src = am_bool & (rank > (lengths.unsqueeze(1) - state_len))
            if valid_src.any():
                b_idx, t_idx = valid_src.nonzero(as_tuple=True)
                dst_idx = rank[b_idx, t_idx] - lengths[b_idx] + (state_len - 1)
                tail_seq[b_idx, dst_idx] = seq_t[b_idx, t_idx]
        else:
            if T >= state_len:
                tail_seq = seq_t[:, -state_len:]
            else:
                tail_seq[:, (state_len - T):] = seq_t
        active = tail_seq[:, -1:]
        keep = (tail_seq == active) & (tail_seq >= 0)
        conv_state.mul_(keep.unsqueeze(1).to(conv_state.dtype))

    def _build_flat_seq_idx(self, valid, lengths, seq_idx, device):
        B = lengths.size(0)
        flat_b = torch.repeat_interleave(torch.arange(B, device=device, dtype=torch.long), lengths)
        if seq_idx is None:
            return flat_b.to(torch.int32).unsqueeze(0).contiguous()
        flat_s = seq_idx[valid].long()
        if flat_s.numel() <= 1:
            return torch.zeros(1, flat_s.numel(), device=device, dtype=torch.int32)
        boundary = (flat_b[1:] != flat_b[:-1]) | (flat_s[1:] != flat_s[:-1])
        out = torch.zeros(flat_s.numel(), device=device, dtype=torch.int32)
        out[1:] = torch.cumsum(boundary.to(torch.int32), dim=0)
        return out.unsqueeze(0).contiguous()

    @staticmethod
    def _iter_contiguous_seq_spans(seq_1d):
        vals = seq_1d.tolist()
        spans = []
        n = len(vals)
        i = 0
        while i < n:
            v = vals[i]
            if v < 0:
                i += 1
                continue
            j = i + 1
            while j < n and vals[j] == v:
                j += 1
            spans.append((i, j))
            i = j
        return spans

    def _run_sequence_split_prefill(self, x, valid, lengths, inference_params, seq_idx):
        B = x.shape[0]
        layer_idx = getattr(self.mamba, "layer_idx", None)
        self._ensure_cache(inference_params, B, x.dtype)
        out = torch.zeros_like(x)
        has_cache = (layer_idx is not None) and (layer_idx in inference_params.key_value_memory_dict)
        for b in range(B):
            L_b = int(lengths[b].item())
            if L_b <= 0:
                continue
            x_b = x[b:b + 1, valid[b], :]
            if seq_idx is not None:
                s_b = seq_idx[b, valid[b]].to(torch.int32)
                spans = self._iter_contiguous_seq_spans(s_b)
            else:
                spans = [(0, L_b)]
            if not spans:
                continue
            y_b = torch.zeros_like(x_b)
            last_idx = len(spans) - 1
            for idx, (st, ed) in enumerate(spans):
                x_seg = x_b[:, st:ed, :]
                if idx < last_idx:
                    y_seg = self.mamba(x_seg)
                else:
                    sub_inf = _create_inference_params(
                        max_seqlen=max(getattr(inference_params, "max_seqlen", ed - st), ed - st),
                        max_batch_size=1,
                        num_heads=getattr(self.mamba, "nheads", 1),
                        head_dim=getattr(self.mamba, "headdim", 64),
                        n_layer=1,
                    )
                    sub_inf.seqlen_offset = 0
                    if has_cache:
                        orig_conv, orig_ssm = inference_params.key_value_memory_dict[layer_idx]
                        sub_conv = torch.zeros(
                            1, orig_conv.shape[2], orig_conv.shape[1],
                            device=orig_conv.device, dtype=orig_conv.dtype,
                        ).transpose(1, 2)
                        sub_ssm = torch.zeros_like(orig_ssm[b:b + 1])
                        sub_inf.key_value_memory_dict[layer_idx] = (sub_conv, sub_ssm)
                    y_seg = self.mamba(x_seg, inference_params=sub_inf)
                    if has_cache and layer_idx in sub_inf.key_value_memory_dict:
                        orig_conv, orig_ssm = inference_params.key_value_memory_dict[layer_idx]
                        sub_conv, sub_ssm = sub_inf.key_value_memory_dict[layer_idx]
                        orig_conv[b:b + 1].copy_(sub_conv)
                        orig_ssm[b:b + 1].copy_(sub_ssm)
                y_b[:, st:ed, :] = y_seg
            out[b:b + 1, valid[b], :] = y_b
        return out

    def _run_sequence_split_training(self, x_seq, seq_arr):
        B, T, D = x_seq.shape
        row_outs = []
        for b in range(B):
            spans = self._iter_contiguous_seq_spans(seq_arr[b])
            if not spans:
                row_outs.append(torch.zeros_like(x_seq[b:b + 1]))
            elif len(spans) == 1 and spans[0] == (0, T):
                row_outs.append(self.mamba(x_seq[b:b + 1]))
            else:
                chunks = []
                cur = 0
                for st, ed in spans:
                    if st > cur:
                        chunks.append(torch.zeros(1, st - cur, D, device=x_seq.device, dtype=x_seq.dtype))
                    chunks.append(self.mamba(x_seq[b:b + 1, st:ed, :]))
                    cur = ed
                if cur < T:
                    chunks.append(torch.zeros(1, T - cur, D, device=x_seq.device, dtype=x_seq.dtype))
                row_outs.append(torch.cat(chunks, dim=1))
        return torch.cat(row_outs, dim=0)

    def _run_per_sample_prefill(self, x, valid, lengths, inference_params, seq_idx):
        if seq_idx is not None and not self._can_use_seq_idx():
            return self._run_sequence_split_prefill(x, valid, lengths, inference_params, seq_idx)
        B = x.shape[0]
        layer_idx = getattr(self.mamba, "layer_idx", None)
        self._ensure_cache(inference_params, B, x.dtype)
        out = torch.zeros_like(x)
        has_cache = (layer_idx is not None) and (layer_idx in inference_params.key_value_memory_dict)
        for b in range(B):
            L_b = int(lengths[b].item())
            if L_b <= 0:
                continue
            x_b = x[b:b + 1, valid[b], :]
            s_b = self._sanitize_seq_idx(seq_idx[b:b + 1, valid[b]]) if seq_idx is not None else None
            sub_inf = _create_inference_params(
                max_seqlen=max(getattr(inference_params, "max_seqlen", L_b), L_b),
                max_batch_size=1,
                num_heads=getattr(self.mamba, "nheads", 1),
                head_dim=getattr(self.mamba, "headdim", 64),
                n_layer=1,
            )
            sub_inf.seqlen_offset = 0
            if has_cache:
                orig_conv, orig_ssm = inference_params.key_value_memory_dict[layer_idx]
                sub_conv = torch.zeros(
                    1, orig_conv.shape[2], orig_conv.shape[1],
                    device=orig_conv.device, dtype=orig_conv.dtype,
                ).transpose(1, 2)
                sub_ssm = torch.zeros_like(orig_ssm[b:b + 1])
                sub_inf.key_value_memory_dict[layer_idx] = (sub_conv, sub_ssm)
            try:
                y_b = self.mamba(x_b, inference_params=sub_inf, seq_idx=s_b)
            except (TypeError, AssertionError):
                return self._run_sequence_split_prefill(x, valid, lengths, inference_params, seq_idx)
            out[b:b + 1, valid[b], :] = y_b
            if has_cache and layer_idx in sub_inf.key_value_memory_dict:
                orig_conv, orig_ssm = inference_params.key_value_memory_dict[layer_idx]
                sub_conv, sub_ssm = sub_inf.key_value_memory_dict[layer_idx]
                orig_conv[b:b + 1].copy_(sub_conv)
                orig_ssm[b:b + 1].copy_(sub_ssm)
        return out

    def forward(self, x, inference_params=None, seq_idx=None, attention_mask=None):
        B, T, D = x.shape
        is_prefill = (inference_params is None) or (getattr(inference_params, "seqlen_offset", 0) == 0)

        if inference_params is not None and is_prefill:
            self._ensure_cache(inference_params, B, x.dtype)

        eff_mask = attention_mask
        if eff_mask is None and seq_idx is not None and is_prefill and T > 0:
            if (seq_idx[:, :T] < 0).any().item():
                eff_mask = (seq_idx[:, :T] >= 0)

        if inference_params is not None and is_prefill and T >= 1 and eff_mask is not None:
            am = eff_mask
            if am.size(1) >= T:
                am = am[:, :T]
            else:
                am = None
            if am is not None:
                valid = am.bool()
                if seq_idx is not None:
                    valid = valid & (seq_idx[:, :T] >= 0)
                lengths = valid.long().sum(dim=1)
                if not (lengths == T).all().item():
                    if (lengths == 0).all().item():
                        return torch.zeros_like(x)
                    if T == 1:
                        nz = (lengths > 0).nonzero(as_tuple=True)[0]
                        layer_idx = getattr(self.mamba, "layer_idx", None)
                        sub_inf = _create_inference_params(
                            max_seqlen=max(getattr(inference_params, "max_seqlen", 1), 1),
                            max_batch_size=int(nz.numel()),
                            num_heads=getattr(self.mamba, "nheads", 1),
                            head_dim=getattr(self.mamba, "headdim", 64),
                            n_layer=1,
                        )
                        sub_inf.seqlen_offset = 0
                        if layer_idx is not None and layer_idx in inference_params.key_value_memory_dict:
                            orig_conv, orig_ssm = inference_params.key_value_memory_dict[layer_idx]
                            sub_conv = orig_conv[nz].clone()
                            sub_ssm = orig_ssm[nz].clone()
                            sub_inf.key_value_memory_dict[layer_idx] = (sub_conv, sub_ssm)
                        out_nz = self.mamba(x[nz], inference_params=sub_inf, seq_idx=None)
                        if layer_idx is not None and layer_idx in inference_params.key_value_memory_dict:
                            orig_conv, orig_ssm = inference_params.key_value_memory_dict[layer_idx]
                            sub_conv, sub_ssm = sub_inf.key_value_memory_dict[layer_idx]
                            orig_conv[nz] = sub_conv
                            orig_ssm[nz] = sub_ssm
                        out = torch.zeros_like(x)
                        out[nz] = out_nz
                        self._mask_conv_state_seq_boundary(inference_params, seq_idx, am, T)
                        return out
                    if (lengths > 0).all().item():
                        if self._can_use_varlen():
                            x_flat = x[valid].unsqueeze(0)
                            cu = torch.cat([
                                torch.zeros(1, device=x.device, dtype=torch.int32),
                                torch.cumsum(lengths, dim=0).to(torch.int32),
                            ])
                            seq_idx_flat = self._build_flat_seq_idx(valid, lengths, seq_idx, x.device)
                            try:
                                out_flat = self.mamba(
                                    x_flat,
                                    inference_params=inference_params,
                                    cu_seqlens=cu,
                                    seq_idx=seq_idx_flat,
                                )
                                out = torch.zeros_like(x)
                                out[valid] = out_flat.squeeze(0)
                                self._mask_conv_state_seq_boundary(inference_params, seq_idx, am, T)
                                return out
                            except (TypeError, AssertionError):
                                pass
                        out = self._run_per_sample_prefill(x, valid, lengths, inference_params, seq_idx)
                        self._mask_conv_state_seq_boundary(inference_params, seq_idx, am, T)
                        return out
                    else:
                        if self._can_use_varlen():
                            nz = (lengths > 0).nonzero(as_tuple=True)[0]
                            nz_lengths = lengths[nz]
                            nz_valid = valid[nz]
                            x_flat = x[nz][nz_valid].unsqueeze(0)
                            cu = torch.cat([
                                torch.zeros(1, device=x.device, dtype=torch.int32),
                                torch.cumsum(nz_lengths, dim=0).to(torch.int32),
                            ])
                            seq_idx_nz = seq_idx[nz] if seq_idx is not None else None
                            seq_idx_flat = self._build_flat_seq_idx(nz_valid, nz_lengths, seq_idx_nz, x.device)
                            layer_idx = getattr(self.mamba, "layer_idx", None)
                            sub_inf = _create_inference_params(
                                max_seqlen=getattr(inference_params, "max_seqlen", T),
                                max_batch_size=int(nz.numel()),
                                num_heads=getattr(self.mamba, "nheads", 1),
                                head_dim=getattr(self.mamba, "headdim", 64),
                                n_layer=1,
                            )
                            sub_inf.seqlen_offset = 0
                            if layer_idx is not None and layer_idx in inference_params.key_value_memory_dict:
                                orig_conv, orig_ssm = inference_params.key_value_memory_dict[layer_idx]
                                sub_conv = orig_conv[nz].clone()
                                sub_ssm = orig_ssm[nz].clone()
                                sub_inf.key_value_memory_dict[layer_idx] = (sub_conv, sub_ssm)
                            try:
                                out_flat = self.mamba(
                                    x_flat,
                                    inference_params=sub_inf,
                                    cu_seqlens=cu,
                                    seq_idx=seq_idx_flat,
                                )
                                if layer_idx is not None and layer_idx in inference_params.key_value_memory_dict:
                                    orig_conv, orig_ssm = inference_params.key_value_memory_dict[layer_idx]
                                    sub_conv, sub_ssm = sub_inf.key_value_memory_dict[layer_idx]
                                    orig_conv[nz] = sub_conv
                                    orig_ssm[nz] = sub_ssm
                                out = torch.zeros_like(x)
                                out[valid] = out_flat.squeeze(0)
                                self._mask_conv_state_seq_boundary(inference_params, seq_idx, am, T)
                                return out
                            except (TypeError, AssertionError):
                                pass
                        out = self._run_per_sample_prefill(x, valid, lengths, inference_params, seq_idx)
                        self._mask_conv_state_seq_boundary(inference_params, seq_idx, am, T)
                        return out

        if inference_params is None and eff_mask is not None and eff_mask.size(1) >= T and T > 0:
            am_bool = eff_mask[:, :T].bool()
            if seq_idx is not None:
                am_bool = am_bool & (seq_idx[:, :T] >= 0)
            am_f = am_bool.unsqueeze(-1).to(x.dtype)
            x_in = x * am_f
            real_pos = am_bool.long().cumsum(dim=1) - 1
            pos_safe = torch.where(am_bool, real_pos, torch.full_like(real_pos, T))
            x_comp = torch.zeros(B, T + 1, D, device=x.device, dtype=x.dtype).scatter(
                1, pos_safe.unsqueeze(-1).expand(-1, -1, D), x_in)[:, :T, :]
            if seq_idx is not None:
                seq_comp = torch.full((B, T + 1), -1, device=seq_idx.device, dtype=torch.int32).scatter(
                    1, pos_safe, seq_idx[:, :T].to(torch.int32))[:, :T].contiguous()
                if self._can_use_seq_idx():
                    seq_comp_safe = self._sanitize_seq_idx(seq_comp)
                    try:
                        y_comp = self.mamba(x_comp, seq_idx=seq_comp_safe)
                    except (TypeError, AssertionError):
                        y_comp = self._run_sequence_split_training(x_comp, seq_comp)
                else:
                    y_comp = self._run_sequence_split_training(x_comp, seq_comp)
            else:
                y_comp = self.mamba(x_comp)
            out = F.pad(y_comp, (0, 0, 0, 1)).gather(1, pos_safe.unsqueeze(-1).expand(-1, -1, D))
            return out * am_f

        if inference_params is not None:
            if not is_prefill:
                am_cur = attention_mask[:, -T:].bool() if (attention_mask is not None and attention_mask.size(1) >= T) else None
                if T == 1:
                    vm = am_cur[:, 0] if am_cur is not None else None
                    return self._step_single_token(x, inference_params, valid_mask=vm)
                outs = []
                for t in range(T):
                    vm = am_cur[:, t] if am_cur is not None else None
                    outs.append(self._step_single_token(x[:, t:t + 1, :], inference_params, valid_mask=vm))
                return torch.cat(outs, dim=1)
            if seq_idx is not None:
                if T == 1:
                    out = self.mamba(x, inference_params=inference_params, seq_idx=None)
                    self._mask_conv_state_seq_boundary(inference_params, seq_idx, attention_mask, T)
                    return out
                if self._can_use_seq_idx():
                    seq_arg = self._sanitize_seq_idx(seq_idx[:, :T])
                    try:
                        out = self.mamba(x, inference_params=inference_params, seq_idx=seq_arg)
                        self._mask_conv_state_seq_boundary(inference_params, seq_idx, attention_mask, T)
                        return out
                    except (TypeError, AssertionError):
                        pass
                valid_all = torch.ones(B, T, dtype=torch.bool, device=x.device)
                lengths_all = torch.full((B,), T, dtype=torch.long, device=x.device)
                return self._run_sequence_split_prefill(x, valid_all, lengths_all, inference_params, seq_idx)
            return self.mamba(x, inference_params=inference_params)
        if seq_idx is not None:
            if self._can_use_seq_idx():
                try:
                    return self.mamba(x, seq_idx=self._sanitize_seq_idx(seq_idx[:, :T]))
                except (TypeError, AssertionError):
                    pass
            return self._run_sequence_split_training(x, seq_idx[:, :T].to(torch.int32))
        return self.mamba(x)


class Engram(nn.Module):
    def __init__(self, d_model=640, num_heads=4, table_size=50000, mem_dim=64,
                 max_ngram=3):
        super().__init__()
        self.num_heads = num_heads
        self.mem_dim = mem_dim
        self.max_ngram = max_ngram
        self.tables = nn.ModuleDict()
        for n in range(2, max_ngram + 1):
            for h in range(num_heads):
                prime_size = self._next_prime(table_size + h * 997 + n * 7919)
                self.tables[f"n{n}_h{h}"] = nn.Embedding(prime_size, mem_dim)
        self.mem_proj = nn.Linear(mem_dim, d_model)
        self.gate = nn.Linear(d_model + d_model, 1)

    def _next_prime(self, n):
        def is_prime(x):
            if x < 2:
                return False
            for i in range(2, int(x ** 0.5) + 1):
                if x % i == 0:
                    return False
            return True
        while not is_prime(n):
            n += 1
        return n

    def _poly_hash(self, ngram_ids, table_size):
        base = 31
        ids = ngram_ids.long()
        h = torch.zeros_like(ids[..., 0])
        for i in range(ids.size(-1)):
            h = (h * base + ids[..., i]) % table_size
        return h

    def forward(self, x, input_ids, T_out=None, seq_idx=None,
                attention_mask=None, is_decode=False):
        B, T_in, D = x.shape
        T_out = T_out if T_out is not None else T_in
        T_ng = input_ids.shape[1]
        use_compact = (not is_decode) and (attention_mask is not None) and (T_ng > 0)
        if use_compact:
            am_bool = attention_mask[:, :T_ng].bool()
            real_pos = am_bool.long().cumsum(dim=1) - 1
            pos_safe = torch.where(am_bool, real_pos, torch.full_like(real_pos, T_ng))
            ids_eff = torch.full((B, T_ng + 1), -1, device=input_ids.device, dtype=input_ids.dtype).scatter(
                1, pos_safe, input_ids)[:, :T_ng]
            seq_eff = (
                torch.full((B, T_ng + 1), -1, device=seq_idx.device, dtype=seq_idx.dtype).scatter(
                    1, pos_safe, seq_idx[:, :T_ng])[:, :T_ng]
                if seq_idx is not None else None
            )
            T_work = T_ng
        else:
            ids_eff = input_ids
            seq_eff = seq_idx
            T_work = T_out

        mem = torch.zeros(B, T_work, self.mem_dim, device=x.device, dtype=torch.float32)
        cnt = torch.zeros(B, T_work, 1, device=x.device, dtype=torch.float32)
        for n in range(2, self.max_ngram + 1):
            if T_ng < n:
                continue
            if is_decode:
                ngrams = ids_eff[:, -n:].unsqueeze(1)
                valid_dec = (ngrams >= 0).all(dim=-1)
                ngrams_safe = ngrams.clamp(min=0)
            else:
                ngrams = torch.stack([ids_eff[:, i:T_ng - n + 1 + i] for i in range(n)], dim=-1)
                ngrams_safe = ngrams.clamp(min=0)
                if seq_eff is not None or use_compact:
                    valid = (ngrams >= 0).all(dim=-1)
                    if seq_eff is not None:
                        ref = seq_eff[:, :T_ng - n + 1]
                        for i in range(n):
                            s_i = seq_eff[:, i:T_ng - n + 1 + i]
                            valid = valid & (s_i == ref) & (s_i >= 0)
                else:
                    valid = None
            for h in range(self.num_heads):
                table = self.tables[f"n{n}_h{h}"]
                hv = self._poly_hash(ngrams_safe, table.num_embeddings)
                if is_decode:
                    emb = table(hv).float()
                    v_f = valid_dec.to(torch.float32).unsqueeze(-1)
                    mem[:, -1:, :] += emb * v_f
                    cnt[:, -1:, :] += v_f
                else:
                    emb = table(hv).float()
                    if valid is not None:
                        v_f = valid.to(torch.float32).unsqueeze(-1)
                        mem[:, n - 1:, :] += emb * v_f
                        cnt[:, n - 1:, :] += v_f
                    else:
                        mem[:, n - 1:, :] += emb
                        cnt[:, n - 1:, :] += 1.0
        mem = mem / cnt.clamp(min=1.0)
        if use_compact:
            mem = F.pad(mem, (0, 0, 0, 1)).gather(
                1, pos_safe[:, :T_out].unsqueeze(-1).expand(-1, -1, self.mem_dim))
        mem = mem.to(x.dtype)
        mem = self.mem_proj(mem)
        gate = torch.sigmoid(self.gate(torch.cat([x, mem], dim=-1)))
        return x + gate * mem


class TransformerFFN(nn.Module):
    def __init__(self, d_model=1024, ratio=4):
        super().__init__()
        hidden = d_model * ratio
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(x + self.down(F.silu(self.gate(x)) * self.up(x)))

class HyperConnection(nn.Module):
    def __init__(self, num_streams=16, d_model=640, top_k=4):
        super().__init__()
        self.num_streams = num_streams
        self.top_k = min(top_k, num_streams)
        self.in_proj = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(num_streams)])
        self.out_proj = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(num_streams)])
        with torch.no_grad():
            for i in range(num_streams):
                nn.init.eye_(self.in_proj[i].weight)
                nn.init.eye_(self.out_proj[i].weight)
                if i > 0:
                    self.in_proj[i].weight.data += torch.randn_like(self.in_proj[i].weight) * 0.02
                    self.out_proj[i].weight.data += torch.randn_like(self.out_proj[i].weight) * 0.02
        self.stream_weights = nn.Parameter(torch.zeros(num_streams))
        hidden_router = max(d_model // 4, 64)
        self.router = nn.Sequential(
            nn.Linear(d_model * 2, hidden_router),
            nn.SiLU(),
            nn.Linear(hidden_router, 1),
        )

    def _route_and_combine(self, stacked, x_out, attention_mask, seq_idx):
        n = self.num_streams
        B, _, T, D = stacked.shape
        if (isinstance(self.router, nn.Sequential) and len(self.router) == 3
                and isinstance(self.router[0], nn.Linear) and isinstance(self.router[2], nn.Linear)):
            r0 = self.router[0]
            r2 = self.router[2]
            dt = stacked.dtype
            W1_s = r0.weight[:, :D].to(dt)
            W1_x = r0.weight[:, D:].to(dt)
            b1 = r0.bias.to(dt) if r0.bias is not None else None
            h_s = F.linear(stacked, W1_s)
            h_x = F.linear(x_out.to(dt), W1_x, b1).unsqueeze(1)
            r_hid = self.router[1](h_s + h_x)
            r_out = F.linear(r_hid, r2.weight.to(dt), r2.bias.to(dt) if r2.bias is not None else None)
        else:
            x_out_e = x_out.unsqueeze(1).expand(-1, n, -1, -1)
            combined = torch.cat([stacked, x_out_e], dim=-1)
            r_out = self.router(combined)

        r_tok = r_out.squeeze(-1)
        device = x_out.device
        valid_tok = torch.ones(B, T, dtype=torch.bool, device=device)
        if attention_mask is not None:
            valid_tok = valid_tok & attention_mask[:, :T].bool()
        if seq_idx is not None:
            seq_t = seq_idx[:, :T].to(torch.int32)
            valid_tok = valid_tok & (seq_t >= 0)
        else:
            seq_t = None

        v_f = valid_tok.to(dtype=r_tok.dtype)
        r_masked = r_tok * v_f.unsqueeze(1)
        r_cs = r_masked.cumsum(dim=2)
        cnt_cs = v_f.cumsum(dim=1)

        if seq_t is not None:
            seq_eff = torch.where(valid_tok, seq_t, torch.full_like(seq_t, -1))
            idx_t = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
            last_v_idx = torch.where(valid_tok, idx_t, torch.zeros_like(idx_t)).cummax(dim=1).values
            ff_seq = seq_eff.gather(1, last_v_idx)
            prev_ff = F.pad(ff_seq[:, :-1], (1, 0), value=-999999)
            is_seq_start = valid_tok & (ff_seq != prev_ff)
            start_idx = torch.where(is_seq_start, idx_t, torch.zeros_like(idx_t)).cummax(dim=1).values
            r_prev = F.pad(r_cs[:, :, :-1], (1, 0), value=0.0)
            cnt_prev_cs = F.pad(cnt_cs[:, :-1], (1, 0), value=0.0)
            base_r = r_prev.gather(2, start_idx.unsqueeze(1).expand(-1, n, -1))
            base_cnt = cnt_prev_cs.gather(1, start_idx)
            r_sum_seq = r_cs - base_r
            cnt_seq = cnt_cs - base_cnt
            scores_tok = r_sum_seq / cnt_seq.clamp(min=1.0).unsqueeze(1)
        else:
            scores_tok = r_cs / cnt_cs.clamp(min=1.0).unsqueeze(1)

        topk_vals, topk_idx = scores_tok.topk(self.top_k, dim=1)
        topk_gates = torch.sigmoid(topk_vals.float()).to(x_out.dtype)
        tok_stream_mask = torch.zeros(B, n, T, device=x_out.device, dtype=x_out.dtype).scatter(
            1, topk_idx, topk_gates
        )
        tok_stream_mask = tok_stream_mask * valid_tok.to(dtype=x_out.dtype).unsqueeze(1)

        W_out = torch.stack([p.weight for p in self.out_proj], dim=0).to(x_out.dtype)
        deltas_all = torch.matmul(x_out.unsqueeze(1), W_out.unsqueeze(0).transpose(-1, -2))
        return stacked + tok_stream_mask.unsqueeze(-1) * deltas_all

    def forward(self, streams, layer_fn, attention_mask=None, seq_idx=None, num_seqs=None,
                router_state=None, return_router_state=False):
        n = self.num_streams
        w = F.softmax(self.stream_weights, dim=0)
        stacked = torch.stack(streams, dim=1)
        B, _, T, D = stacked.shape
        w_cast = w.to(stacked.dtype)
        W_in = torch.stack([p.weight for p in self.in_proj], dim=0).to(stacked.dtype)
        w_streams = (stacked * w_cast.view(1, n, 1, 1)).permute(1, 0, 2, 3).reshape(n, B * T, D)
        x_in = torch.bmm(w_streams, W_in.transpose(-1, -2)).sum(dim=0).view(B, T, D)
        x_out = layer_fn(x_in)

        if T == 0:
            new_streams = [stacked[:, i] for i in range(n)]
            if return_router_state:
                return new_streams, router_state
            return new_streams

        if self.training and torch.is_grad_enabled() and router_state is None and not return_router_state:
            new_stacked = checkpoint(
                self._route_and_combine, stacked, x_out, attention_mask, seq_idx,
                use_reentrant=False
            )
            return [new_stacked[:, i] for i in range(n)]

        if (isinstance(self.router, nn.Sequential) and len(self.router) == 3
                and isinstance(self.router[0], nn.Linear) and isinstance(self.router[2], nn.Linear)):
            r0 = self.router[0]
            r2 = self.router[2]
            dt = stacked.dtype
            W1_s = r0.weight[:, :D].to(dt)
            W1_x = r0.weight[:, D:].to(dt)
            b1 = r0.bias.to(dt) if r0.bias is not None else None
            h_s = F.linear(stacked, W1_s)
            h_x = F.linear(x_out.to(dt), W1_x, b1).unsqueeze(1)
            r_hid = self.router[1](h_s + h_x)
            r_out = F.linear(r_hid, r2.weight.to(dt), r2.bias.to(dt) if r2.bias is not None else None)
        else:
            x_out_e = x_out.unsqueeze(1).expand(-1, n, -1, -1)
            combined = torch.cat([stacked, x_out_e], dim=-1)
            r_out = self.router(combined)

        r_tok = r_out.squeeze(-1)
        device = x_out.device
        new_router_state = None

        if router_state is not None:
            r_prev_sum, cnt_prev = router_state
            if attention_mask is not None:
                am_cur = attention_mask[:, -T:].bool()
                am_f = am_cur.to(dtype=r_tok.dtype)
                r_masked = r_tok * am_f.unsqueeze(1)
                r_sum_tok = r_prev_sum.unsqueeze(2) + r_masked.cumsum(dim=2)
                cnt_tok = cnt_prev.unsqueeze(1) + am_f.cumsum(dim=1)
                valid_tok = am_cur
            else:
                r_sum_tok = r_prev_sum.unsqueeze(2) + r_tok.cumsum(dim=2)
                cnt_tok = cnt_prev.unsqueeze(1) + torch.arange(1, T + 1, device=device, dtype=r_tok.dtype).unsqueeze(0)
                valid_tok = None
            scores_tok = r_sum_tok / cnt_tok.clamp(min=1.0).unsqueeze(1)
            if return_router_state:
                new_router_state = (r_sum_tok[:, :, -1].detach(), cnt_tok[:, -1].detach())
        else:
            valid_tok = torch.ones(B, T, dtype=torch.bool, device=device)
            if attention_mask is not None:
                valid_tok = valid_tok & attention_mask[:, :T].bool()
            if seq_idx is not None:
                seq_t = seq_idx[:, :T].to(torch.int32)
                valid_tok = valid_tok & (seq_t >= 0)
            else:
                seq_t = None

            v_f = valid_tok.to(dtype=r_tok.dtype)
            r_masked = r_tok * v_f.unsqueeze(1)
            r_cs = r_masked.cumsum(dim=2)
            cnt_cs = v_f.cumsum(dim=1)

            if seq_t is not None:
                seq_eff = torch.where(valid_tok, seq_t, torch.full_like(seq_t, -1))
                idx_t = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
                last_v_idx = torch.where(valid_tok, idx_t, torch.zeros_like(idx_t)).cummax(dim=1).values
                ff_seq = seq_eff.gather(1, last_v_idx)
                prev_ff = F.pad(ff_seq[:, :-1], (1, 0), value=-999999)
                is_seq_start = valid_tok & (ff_seq != prev_ff)
                start_idx = torch.where(is_seq_start, idx_t, torch.zeros_like(idx_t)).cummax(dim=1).values
                r_prev = F.pad(r_cs[:, :, :-1], (1, 0), value=0.0)
                cnt_prev_cs = F.pad(cnt_cs[:, :-1], (1, 0), value=0.0)
                base_r = r_prev.gather(2, start_idx.unsqueeze(1).expand(-1, n, -1))
                base_cnt = cnt_prev_cs.gather(1, start_idx)
                r_sum_seq = r_cs - base_r
                cnt_seq = cnt_cs - base_cnt
                scores_tok = r_sum_seq / cnt_seq.clamp(min=1.0).unsqueeze(1)
                if return_router_state:
                    has_v = valid_tok.any(dim=1)
                    last_pos = torch.where(valid_tok, idx_t, torch.zeros_like(idx_t)).max(dim=1).values
                    b_all = torch.arange(B, device=device)
                    r_last = torch.where(has_v.unsqueeze(1), r_sum_seq[b_all, :, last_pos], torch.zeros(B, n, device=device, dtype=r_tok.dtype))
                    cnt_last = torch.where(has_v, cnt_seq[b_all, last_pos], torch.zeros(B, device=device, dtype=r_tok.dtype))
                    new_router_state = (r_last.detach(), cnt_last.detach())
            else:
                scores_tok = r_cs / cnt_cs.clamp(min=1.0).unsqueeze(1)
                if return_router_state:
                    new_router_state = (r_cs[:, :, -1].detach(), cnt_cs[:, -1].detach())

        topk_vals, topk_idx = scores_tok.topk(self.top_k, dim=1)
        topk_gates = torch.sigmoid(topk_vals.float()).to(x_out.dtype)
        tok_stream_mask = torch.zeros(B, n, T, device=x_out.device, dtype=x_out.dtype).scatter(
            1, topk_idx, topk_gates
        )
        if valid_tok is not None:
            tok_stream_mask = tok_stream_mask * valid_tok.to(dtype=x_out.dtype).unsqueeze(1)

        W_out = torch.stack([p.weight for p in self.out_proj], dim=0).to(x_out.dtype)
        deltas_all = torch.matmul(x_out.unsqueeze(1), W_out.unsqueeze(0).transpose(-1, -2))
        new_stacked = stacked + tok_stream_mask.unsqueeze(-1) * deltas_all
        new_streams = [new_stacked[:, i] for i in range(n)]
        if return_router_state:
            return new_streams, new_router_state
        return new_streams

class BlockRouter:
    def __init__(self, top_k_a=4, top_k_b=4):
        self.top_k_a = top_k_a
        self.top_k_b = top_k_b

    def __call__(self, C, block_size, num_blocks, attention_mask=None):
        B, T = C.shape
        pad = num_blocks * block_size - T
        C_p = F.pad(C.float(), (0, pad), value=0.0) if pad > 0 else C.float()
        C_blocks = C_p.view(B, num_blocks, block_size)
        if attention_mask is not None:
            am = attention_mask[:, :T].float()
        else:
            am = torch.ones(B, T, device=C.device, dtype=torch.float32)
        if pad > 0:
            am = F.pad(am, (0, pad), value=0.0)
        am_blocks = am.view(B, num_blocks, block_size)
        raw_counts = am_blocks.sum(-1)
        valid = raw_counts > 0
        counts = raw_counts.clamp(min=1.0)
        C_masked = C_blocks * am_blocks
        score_a = C_masked.max(dim=-1).values
        score_b = C_masked.sum(dim=-1) / counts
        score_a = score_a.masked_fill(~valid, float('-inf'))
        score_b = score_b.masked_fill(~valid, float('-inf'))
        k_a = min(self.top_k_a, num_blocks)
        k_b = min(self.top_k_b, num_blocks)
        _, idx_a = score_a.topk(k_a, dim=-1)
        _, idx_b = score_b.topk(k_b, dim=-1)
        mask_a = torch.zeros(B, num_blocks, dtype=torch.bool, device=C.device)
        mask_b = torch.zeros(B, num_blocks, dtype=torch.bool, device=C.device)
        mask_a.scatter_(1, idx_a, True)
        mask_b.scatter_(1, idx_b, True)
        selected = (mask_a | mask_b) & valid
        advanced = (mask_a & mask_b) & valid
        return selected, advanced

    def route_causal_from_scores(self, score_a, score_b, valid_blk):
        B, T, num_blocks = score_a.shape
        device = score_a.device
        k_a = min(self.top_k_a, num_blocks)
        k_b = min(self.top_k_b, num_blocks)
        _, idx_a = score_a.topk(k_a, dim=-1)
        _, idx_b = score_b.topk(k_b, dim=-1)
        mask_a = torch.zeros(B, T, num_blocks, dtype=torch.bool, device=device)
        mask_b = torch.zeros(B, T, num_blocks, dtype=torch.bool, device=device)
        mask_a.scatter_(2, idx_a, True)
        mask_b.scatter_(2, idx_b, True)
        selected = (mask_a | mask_b) & valid_blk
        advanced = (mask_a & mask_b) & valid_blk
        return selected, advanced

    def route_packed(self, C, block_size, num_blocks, seq_idx, num_seqs, attention_mask=None):
        B, T = C.shape
        device = C.device
        pad = num_blocks * block_size - T
        C_p = F.pad(C.float(), (0, pad), value=0.0) if pad > 0 else C.float()
        seq_p = F.pad(seq_idx[:, :T].to(torch.int32), (0, pad), value=-1) if pad > 0 else seq_idx[:, :T].to(torch.int32)
        if attention_mask is not None:
            am = attention_mask[:, :T].float()
            if pad > 0:
                am = F.pad(am, (0, pad), value=0.0)
        else:
            am = torch.ones(B, T, device=device, dtype=torch.float32)
            if pad > 0:
                am = F.pad(am, (0, pad), value=0.0)
        C_blocks = C_p.view(B, 1, num_blocks, block_size)
        seq_blocks = seq_p.view(B, 1, num_blocks, block_size)
        am_blocks = (am.view(B, 1, num_blocks, block_size) > 0)
        seq_range = torch.arange(1, num_seqs + 1, device=device, dtype=torch.int32).view(1, num_seqs, 1, 1)
        m_s = (seq_blocks == seq_range) & am_blocks
        m_sf = m_s.float()
        cnt_s = m_sf.sum(-1)
        valid_s = cnt_s > 0
        C_ms = C_blocks * m_sf
        score_a = C_ms.max(dim=-1).values.masked_fill(~valid_s, float('-inf'))
        score_b = (C_ms.sum(dim=-1) / cnt_s.clamp(min=1.0)).masked_fill(~valid_s, float('-inf'))
        k_a = min(self.top_k_a, num_blocks)
        k_b = min(self.top_k_b, num_blocks)
        _, idx_a = score_a.topk(k_a, dim=-1)
        _, idx_b = score_b.topk(k_b, dim=-1)
        mask_a = torch.zeros(B, num_seqs, num_blocks, dtype=torch.bool, device=device)
        mask_b = torch.zeros(B, num_seqs, num_blocks, dtype=torch.bool, device=device)
        mask_a.scatter_(2, idx_a, True)
        mask_b.scatter_(2, idx_b, True)
        selected_s = (mask_a | mask_b) & valid_s
        advanced_s = (mask_a & mask_b) & valid_s
        return selected_s, advanced_s


class GlobalBlock(nn.Module):
    def __init__(self, d_model, d_model_thought, num_heads, num_kv_heads, head_dim,
                 window_size, use_attention_checkpoint=False, ffn_ratio=4):
        super().__init__()
        self.use_attention_checkpoint = use_attention_checkpoint
        self.conv = CausalConv1d(d_model)
        self.attn = GQAAttention(d_model, num_heads, num_kv_heads, head_dim, window_size)
        self.up_proj = nn.Linear(d_model, d_model_thought)
        self.ffn = TransformerFFN(d_model_thought, ratio=ffn_ratio)
        self.down_proj = nn.Linear(d_model_thought, d_model)

    def forward(self, x, past_kv=None, conv_cache=None, seq_idx=None,
                attention_mask=None, return_importance=False,
                conv_seq_cache=None, past_seq_idx=None, return_seq_cache=False,
                q_pos=None, past_kv_pos=None, routing_spec=None):
        if return_seq_cache:
            x_conv, new_conv, new_conv_seq = self.conv(
                x, conv_cache, seq_idx=seq_idx, attention_mask=attention_mask,
                seq_cache=conv_seq_cache, return_seq_cache=True)
        else:
            x_conv, new_conv = self.conv(
                x, conv_cache, seq_idx=seq_idx, attention_mask=attention_mask,
                seq_cache=conv_seq_cache, return_seq_cache=False)
            new_conv_seq = None
        if self.use_attention_checkpoint and self.training and torch.is_grad_enabled() and past_kv is None:
            if return_importance:
                if routing_spec is not None:
                    def _attn_fn(inp):
                        out, _, c_tup = self.attn._core(
                            inp, None, seq_idx, attention_mask, True, None, q_pos, None,
                            routing_spec=routing_spec)
                        return out, c_tup[0], c_tup[1], c_tup[2], c_tup[3]
                    x_attn, sa, sb, vb, c_last = checkpoint(_attn_fn, x_conv, use_reentrant=True)
                    C = (sa, sb, vb, c_last)
                    new_kv = None
                else:
                    def _attn_fn(inp):
                        out, _, c_imp = self.attn._core(
                            inp, None, seq_idx, attention_mask, True, None, q_pos, None)
                        return out, c_imp
                    x_attn, C = checkpoint(_attn_fn, x_conv, use_reentrant=True)
                    new_kv = None
            else:
                def _attn_fn(inp):
                    return self.attn._core(
                        inp, None, seq_idx, attention_mask, False, None, q_pos, None)[0]
                x_attn = checkpoint(_attn_fn, x_conv, use_reentrant=True)
                new_kv = None
                C = None
        else:
            x_attn, new_kv, C = self.attn(x_conv, past_kv, seq_idx,
                                           attention_mask=attention_mask,
                                           return_importance=return_importance,
                                           past_seq_idx=past_seq_idx,
                                           q_pos=q_pos,
                                           past_kv_pos=past_kv_pos,
                                           routing_spec=routing_spec)
        x_up = self.up_proj(x_attn)
        x_ffn = self.ffn(x_up)
        x_out = self.down_proj(x_ffn)
        if return_seq_cache:
            return x_out, new_kv, new_conv, C, new_conv_seq
        return x_out, new_kv, new_conv, C


class ThoughtBlock(nn.Module):
    def __init__(self, d_model, d_model_thought, num_heads, num_kv_heads, head_dim,
                 block_size, top_k_heads, thought_layer, max_blocks, max_full,
                 mamba_d_state=128, mamba_d_conv=4, mamba_expand=2, mamba_headdim=64,
                 ssm_cache_dtype=None, conv_cache_dtype=None,
                 use_attention_checkpoint=False):
        super().__init__()
        self.use_attention_checkpoint = use_attention_checkpoint
        self.conv = CausalConv1d(d_model)
        self.attn = ThoughtGQAAttention(d_model, num_heads, num_kv_heads, head_dim,
                                          block_size, top_k_heads, max_blocks, max_full)
        self.up_proj = nn.Linear(d_model, d_model_thought)
        self.mamba2 = Mamba2Layer(d_model_thought, d_state=mamba_d_state,
                                    d_conv=mamba_d_conv, expand=mamba_expand,
                                    headdim=mamba_headdim,
                                    ssm_cache_dtype=ssm_cache_dtype,
                                    conv_cache_dtype=conv_cache_dtype)
        self.mamba_gate = nn.Parameter(torch.zeros(1))
        self.thought = thought_layer
        self.down_proj = nn.Linear(d_model_thought, d_model)

    def forward(self, x, cache=None, conv_cache=None, seq_idx=None,
                attention_mask=None, selected=None, advanced=None, reeval=False,
                build_cache=False, mamba_inference_params=None,
                conv_seq_cache=None, cur_seq_ids=None, return_seq_cache=False,
                real_pos=None, selected_per_seq=None, advanced_per_seq=None,
                reeval_mask=None, chunk_mask=None):
        if return_seq_cache:
            x_conv, new_conv, new_conv_seq = self.conv(
                x, conv_cache, seq_idx=seq_idx, attention_mask=attention_mask,
                seq_cache=conv_seq_cache, return_seq_cache=True)
        else:
            x_conv, new_conv = self.conv(
                x, conv_cache, seq_idx=seq_idx, attention_mask=attention_mask,
                seq_cache=conv_seq_cache, return_seq_cache=False)
            new_conv_seq = None
        if cache is None:
            if self.use_attention_checkpoint and self.training and torch.is_grad_enabled() and not build_cache:
                def _attn_fn(inp):
                    return self.attn._core_prefill(
                        inp, seq_idx, attention_mask, selected, advanced, False,
                        real_pos=real_pos, selected_per_seq=selected_per_seq,
                        advanced_per_seq=advanced_per_seq)[0]
                x_attn = checkpoint(_attn_fn, x_conv, use_reentrant=True)
                new_attn_cache = None
            else:
                x_attn, new_attn_cache = self.attn.forward_prefill(
                    x_conv, seq_idx=seq_idx, attention_mask=attention_mask,
                    selected=selected, advanced=advanced, build_cache=build_cache,
                    real_pos=real_pos, selected_per_seq=selected_per_seq,
                    advanced_per_seq=advanced_per_seq)
        else:
            x_attn, new_attn_cache = self.attn.forward_decode(
                x_conv, cache, selected=selected, advanced=advanced,
                reeval=reeval, cur_seq_ids=cur_seq_ids,
                reeval_mask=reeval_mask, chunk_mask=chunk_mask)
        x_up = self.up_proj(x_attn)

        x_mamba = self.mamba2(x_up, inference_params=mamba_inference_params,
                              seq_idx=seq_idx, attention_mask=attention_mask)
        gate = torch.sigmoid(self.mamba_gate)
        x_comb = x_up + gate * x_mamba

        x_th = self.thought(x_comb)
        x_out = self.down_proj(x_th)
        if return_seq_cache:
            return x_out, new_attn_cache, new_conv, new_conv_seq
        return x_out, new_attn_cache, new_conv


class CouncilBackbone(nn.Module):
    def __init__(self, d_model=640, d_model_thought=1024, num_blocks=7, num_streams=16,
                 vocab_size=50257, use_checkpoint=False, window_size=4096,
                 attention_checkpoint=None, config=None):
        super().__init__()
        self.d_model = d_model
        self.d_model_thought = d_model_thought
        self.num_blocks = num_blocks
        self.num_streams = num_streams
        self.use_checkpoint = use_checkpoint
        self.attention_checkpoint = attention_checkpoint if attention_checkpoint is not None else use_checkpoint
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.engram = Engram(d_model, table_size=config.engram_table_size,
                              mem_dim=config.engram_mem_dim)
        self.block_size = config.sliding_window // config.sparse_num_blocks
        self.max_blocks = (config.max_seq_len + self.block_size - 1) // self.block_size
        self.max_full = config.adv_top_k_a + config.adv_top_k_b + config.hybrid_max_full_extra

        g_heads = config.global_num_heads
        g_kv = config.global_num_kv_heads
        t_heads = config.thought_num_heads
        t_kv = config.thought_num_kv_heads
        hd = config.head_dim

        self.blocks = nn.ModuleList()
        mamba_layer_count = 0
        ssm_cache_dtype = getattr(config, "mamba_ssm_cache_dtype", None)
        conv_cache_dtype = getattr(config, "mamba_conv_cache_dtype", None)
        ffn_ratio = getattr(config, "ffn_ratio", 4)
        for i in range(num_blocks):
            if i % 2 == 0:
                block = GlobalBlock(d_model, d_model_thought, g_heads, g_kv, hd, window_size,
                                    use_attention_checkpoint=self.attention_checkpoint,
                                    ffn_ratio=ffn_ratio)
            else:
                th = TransformerFFN(d_model_thought, ratio=ffn_ratio)
                block = ThoughtBlock(d_model, d_model_thought, t_heads, t_kv, hd,
                                      self.block_size, config.thought_head_top_k, th,
                                      self.max_blocks, self.max_full,
                                      mamba_d_state=config.mamba_d_state,
                                      mamba_d_conv=config.mamba_d_conv,
                                      mamba_expand=config.mamba_expand,
                                      mamba_headdim=config.mamba_headdim,
                                      ssm_cache_dtype=ssm_cache_dtype,
                                      conv_cache_dtype=conv_cache_dtype,
                                      use_attention_checkpoint=self.attention_checkpoint)
                block.mamba2.mamba.layer_idx = mamba_layer_count
                mamba_layer_count += 1
            self.blocks.append(block)

        self.hcs = nn.ModuleList([HyperConnection(num_streams, d_model, top_k=config.hyper_top_k)
                                    for _ in range(num_blocks)])
        self.final_mix = nn.Parameter(torch.zeros(num_streams))
        self.norm = nn.LayerNorm(d_model)
        self.block_router = BlockRouter(top_k_a=config.adv_top_k_a, top_k_b=config.adv_top_k_b)
        self.reeval_interval = getattr(config, "hybrid_reeval_interval", 128)
        self.max_seqlen = config.max_seq_len
        self.mamba_layer_count = mamba_layer_count
        self.mamba_nheads = (d_model_thought * config.mamba_expand) // config.mamba_headdim
        self.config_mamba_headdim = config.mamba_headdim

    def _build_seq_idx_from_cu_seqlens(self, cu_seqlens, T, attention_mask, device, target_batch=None):
        if cu_seqlens.dim() == 1:
            cu_seqlens = cu_seqlens.unsqueeze(0)
        B_cu, _ = cu_seqlens.shape
        seq_idx = torch.zeros(B_cu, T, dtype=torch.int32, device=device)
        positions = torch.arange(T, device=device)
        for b in range(B_cu):
            row = cu_seqlens[b].to(torch.long)
            valid_vals = row[row > 0]
            if valid_vals.numel() > 0:
                max_end = int(valid_vals.max().item())
                boundaries = valid_vals[valid_vals < min(max_end, T)]
                if boundaries.numel() > 0:
                    boundaries = torch.unique_consecutive(boundaries)
                    counts = (positions.unsqueeze(0) >= boundaries.unsqueeze(1)).sum(0)
                else:
                    counts = torch.zeros(T, dtype=torch.long, device=device)
                row_seq = (1 + counts).to(torch.int32)
                if max_end < T:
                    row_seq = torch.where(positions < max_end, row_seq, torch.full_like(row_seq, -1))
            elif (row >= 0).any() and row.numel() >= 2:
                row_seq = torch.full((T,), -1, dtype=torch.int32, device=device)
            else:
                row_seq = torch.ones(T, dtype=torch.int32, device=device)
            if attention_mask is not None and attention_mask.size(0) == B_cu and attention_mask.size(1) >= T:
                row_seq = torch.where(attention_mask[b, :T].bool(), row_seq, torch.full_like(row_seq, -1))
            seq_idx[b] = row_seq
        if target_batch is not None:
            if B_cu == 1 and target_batch > 1:
                seq_idx = seq_idx.expand(target_batch, -1).clone()
                if attention_mask is not None and attention_mask.size(0) == target_batch and attention_mask.size(1) >= T:
                    seq_idx = torch.where(attention_mask[:, :T].bool(), seq_idx, torch.full_like(seq_idx, -1))
            elif B_cu != target_batch:
                raise ValueError(f"cu_seqlens batch size {B_cu} does not match input batch size {target_batch}")
        return seq_idx

    def _extract_active_seq_ids(self, seq_idx, T):
        B = seq_idx.shape[0]
        device = seq_idx.device
        seq_t = seq_idx[:, :T].to(torch.int32)
        pos_grid = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        valid_seq = seq_t >= 0
        last_valid_pos = torch.where(valid_seq, pos_grid, torch.full_like(pos_grid, -1)).max(dim=1).values.clamp(min=0)
        active = seq_t[torch.arange(B, device=device), last_valid_pos]
        return torch.where(valid_seq.any(dim=1), active, torch.full_like(active, -1))

    def _extract_last_k_tokens(self, input_ids, token_mask, k, prev_last=None):
        B, T = input_ids.shape
        device = input_ids.device
        if k <= 0:
            return input_ids.new_zeros(B, 0)
        if prev_last is not None:
            prev_mask = prev_last >= 0
            combined_ids = torch.cat([prev_last, input_ids], dim=1)
            combined_mask = torch.cat([prev_mask, token_mask.bool()], dim=1)
        else:
            combined_ids = input_ids
            combined_mask = token_mask.bool()
        out = torch.full((B, k), -1, device=device, dtype=input_ids.dtype)
        lengths = combined_mask.long().sum(dim=1)
        rank = combined_mask.long().cumsum(dim=1)
        valid_src = combined_mask & (rank > (lengths.unsqueeze(1) - k))
        if valid_src.any():
            b_idx, t_idx = valid_src.nonzero(as_tuple=True)
            dst_idx = rank[b_idx, t_idx] - lengths[b_idx] + (k - 1)
            out[b_idx, dst_idx] = combined_ids[b_idx, t_idx]
        return out

    def _compact_importance_and_mask(self, C, kv_pos, num_blocks, seq_vals=None, target_seq=None):
        B = C.shape[0]
        max_len = num_blocks * self.block_size
        C_log = torch.zeros(B, max_len, device=C.device, dtype=torch.float32)
        am_log = torch.zeros(B, max_len, device=C.device, dtype=torch.float32)
        seq_log = torch.full((B, max_len), -1, device=C.device, dtype=torch.int32) if seq_vals is not None else None
        if max_len <= 0 or kv_pos.size(1) == 0:
            return (C_log, am_log, seq_log) if seq_vals is not None else (C_log, am_log)
        valid = (kv_pos >= 0) & (kv_pos < max_len)
        if seq_vals is not None and target_seq is not None:
            valid = valid & (seq_vals == target_seq.view(B, 1)) & (target_seq.view(B, 1) >= 0)
        pos_safe = kv_pos.clamp(min=0, max=max_len - 1)
        v_f = valid.float()
        C_log = C_log.scatter_add(1, pos_safe, C.float() * v_f)
        am_log = am_log.scatter_add(1, pos_safe, v_f)
        if seq_vals is not None:
            seq_in = torch.where(valid, seq_vals.to(torch.int32), torch.full_like(seq_vals, -1, dtype=torch.int32))
            seq_log = torch.full((B, max_len + 1), -1, device=C.device, dtype=torch.int32).scatter(
                1, torch.where(valid, pos_safe, torch.full_like(pos_safe, max_len)), seq_in)[:, :max_len]
            return C_log, am_log, seq_log
        return C_log, am_log

    def _align_route_width(self, sel, adv, target_nb):
        cur_nb = sel.shape[-1]
        if cur_nb >= target_nb:
            return sel[..., :target_nb], adv[..., :target_nb]
        pad = target_nb - cur_nb
        return F.pad(sel, (0, pad), value=False), F.pad(adv, (0, pad), value=False)

    def _forward_decode_chunk_stepwise(self, input_ids, cache, return_cache=False,
                                       cu_seqlens=None, attention_mask=None):
        B, T = input_ids.shape
        device = input_ids.device
        past_len = cache.get("seq_len", 0)
        outs = []
        cur_cache = cache
        for t in range(T):
            ids_t = input_ids[:, t:t + 1]
            if attention_mask is not None:
                if attention_mask.size(1) >= past_len + T:
                    am_t = attention_mask[:, :past_len + t + 1]
                elif attention_mask.size(1) >= T:
                    am_t = attention_mask[:, t:t + 1]
                else:
                    am_t = attention_mask[:, -1:]
            else:
                am_t = None
            out_t, cur_cache = self.forward(
                ids_t,
                cache=cur_cache,
                return_cache=True,
                cu_seqlens=cu_seqlens,
                attention_mask=am_t,
            )
            outs.append(out_t)
        x_out = torch.cat(outs, dim=1) if outs else torch.zeros(B, 0, self.d_model, device=device, dtype=self.embedding.weight.dtype)
        return x_out, (cur_cache if return_cache else None)

    def forward(self, input_ids, cache=None, return_cache=False,
                cu_seqlens=None, attention_mask=None):
        B, T = input_ids.shape
        device = input_ids.device
        is_decode = cache is not None
        want_cache = return_cache and not self.training
        k_ng = self.engram.max_ngram - 1

        if is_decode and T > 1:
            return self._forward_decode_chunk_stepwise(
                input_ids, cache, return_cache=return_cache,
                cu_seqlens=cu_seqlens, attention_mask=attention_mask,
            )

        seq_idx = None
        active_seq_ids = None
        num_packed_seqs = 0
        if is_decode:
            if T == 0:
                raise ValueError("input_ids during decode must have non-zero sequence length")
            active_seq_ids = cache.get("active_seq_ids", None)
            if active_seq_ids is not None:
                seq_idx = active_seq_ids.view(B, 1).expand(B, T)
                if attention_mask is not None and attention_mask.size(1) >= T:
                    seq_idx = torch.where(attention_mask[:, -T:].bool(), seq_idx, torch.full_like(seq_idx, -1))
            attn_seq_idxs = list(cache.get("attn_seq_idxs", [None] * self.num_blocks))
            conv_seq_caches = list(cache.get("conv_seq_caches", [None] * self.num_blocks))
            hc_router_states = list(cache.get("hc_router_states", [None] * self.num_blocks))
        else:
            if cu_seqlens is not None:
                seq_idx = self._build_seq_idx_from_cu_seqlens(cu_seqlens, T, attention_mask, device, target_batch=B)
                active_seq_ids = self._extract_active_seq_ids(seq_idx, T)
                num_packed_seqs = int(seq_idx.max().item()) if seq_idx.numel() > 0 else 0
            attn_seq_idxs = [None] * self.num_blocks
            conv_seq_caches = [None] * self.num_blocks
            hc_router_states = [None] * self.num_blocks

        if is_decode:
            T_step = T
            if attention_mask is not None:
                if attention_mask.size(1) < T_step:
                    raise ValueError(f"attention_mask length {attention_mask.size(1)} < decode chunk length {T_step}")
                am_cur = attention_mask[:, -T_step:].bool()
            else:
                am_cur = None

            x = self.embedding(input_ids)
            prev_last_toks = cache.get("last_tokens", None)
            if prev_last_toks is not None and k_ng > 0:
                toks = torch.cat([prev_last_toks, input_ids], dim=1)
                if T_step == 1:
                    x = self.engram(x, toks, T_out=1, is_decode=True)
                else:
                    toks_mask_cur = am_cur if am_cur is not None else torch.ones(B, T_step, dtype=torch.bool, device=device)
                    toks_mask = torch.cat([prev_last_toks >= 0, toks_mask_cur], dim=1)
                    if active_seq_ids is not None:
                        prev_seq = torch.where(prev_last_toks >= 0, active_seq_ids.view(B, 1), torch.full_like(prev_last_toks, -1, dtype=torch.int32))
                        toks_seq = torch.cat([prev_seq, seq_idx], dim=1)
                    else:
                        toks_seq = None
                    x_padded = F.pad(x, (0, 0, k_ng, 0))
                    x = self.engram(x_padded, toks, T_out=k_ng + T_step, seq_idx=toks_seq,
                                    attention_mask=toks_mask, is_decode=False)[:, k_ng:, :]
            if am_cur is not None:
                x = x * am_cur.unsqueeze(-1).to(x.dtype)
            new_last_tokens = self._extract_last_k_tokens(
                input_ids,
                am_cur if am_cur is not None else torch.ones(B, T_step, dtype=torch.bool, device=device),
                k_ng,
                prev_last=prev_last_toks,
            )

            attn_kvs = list(cache["attn_kvs"])
            conv_caches = list(cache["conv_caches"])
            thought_caches = list(cache["thought_caches"])
            mamba_inf = cache.get("mamba_inference_params", None)
            streams = [x.clone() for _ in range(self.num_streams)]
            past_seq_len = cache.get("seq_len", 0)
            hc_mask = am_cur if am_cur is not None else None
            hc_seq_idx = None
            hc_num_seqs = None
            row_seq_lens_before = cache.get(
                "row_seq_lens",
                torch.full((B,), past_seq_len, device=device, dtype=torch.long),
            )
            if am_cur is not None:
                cur_q_pos = torch.where(
                    am_cur,
                    row_seq_lens_before.unsqueeze(1) + am_cur.long().cumsum(dim=1) - 1,
                    torch.full((B, T_step), -1, device=device, dtype=torch.long),
                )
                row_seq_lens_after = row_seq_lens_before + am_cur.long().sum(dim=1)
            else:
                cur_q_pos = row_seq_lens_before.unsqueeze(1) + torch.arange(T_step, device=device).unsqueeze(0)
                row_seq_lens_after = row_seq_lens_before + T_step
            past_kv_pos = cache.get("attn_kv_pos", None)
            if past_kv_pos is None:
                if past_seq_len > 0:
                    past_kv_pos = torch.arange(past_seq_len, device=device).unsqueeze(0).expand(B, past_seq_len)
                else:
                    past_kv_pos = torch.zeros(B, 0, device=device, dtype=torch.long)
            full_kv_pos = torch.cat([past_kv_pos, cur_q_pos], dim=1)
            real_pos = None
        else:
            x = self.embedding(input_ids)
            x = self.engram(x, input_ids, seq_idx=seq_idx,
                              attention_mask=attention_mask, is_decode=False)
            if attention_mask is not None and T > 0:
                x = x * attention_mask[:, :T].unsqueeze(-1).to(x.dtype)
            elif seq_idx is not None and T > 0:
                x = x * (seq_idx[:, :T] >= 0).unsqueeze(-1).to(x.dtype)
            if want_cache:
                if cu_seqlens is not None and active_seq_ids is not None:
                    lt_mask = (seq_idx[:, :T] == active_seq_ids.view(B, 1)) & (active_seq_ids.view(B, 1) >= 0)
                    if attention_mask is not None:
                        lt_mask = lt_mask & attention_mask[:, :T].bool()
                elif attention_mask is not None:
                    lt_mask = attention_mask[:, :T].bool()
                else:
                    lt_mask = torch.ones(B, T, dtype=torch.bool, device=device)
                new_last_tokens = self._extract_last_k_tokens(input_ids, lt_mask, k_ng)
            else:
                new_last_tokens = None

            attn_kvs = [None] * self.num_blocks
            conv_caches = [None] * self.num_blocks
            thought_caches = [None] * self.num_blocks
            if want_cache:
                mamba_inf = _create_inference_params(
                    max_seqlen=max(self.max_seqlen, T), max_batch_size=B,
                    num_heads=self.mamba_nheads,
                    head_dim=self.config_mamba_headdim if hasattr(self, "config_mamba_headdim") else 64,
                    n_layer=self.mamba_layer_count,
                )
                mamba_inf.seqlen_offset = 0
            else:
                mamba_inf = None
            streams = [x.clone() for _ in range(self.num_streams)]
            past_seq_len = 0
            T_step = T
            am_cur = attention_mask[:, :T].bool() if attention_mask is not None else None
            if attention_mask is not None:
                hc_mask = attention_mask[:, :T]
            elif seq_idx is not None:
                hc_mask = (seq_idx[:, :T] >= 0)
            else:
                hc_mask = None
            hc_seq_idx = seq_idx[:, :T] if (seq_idx is not None and num_packed_seqs > 1) else None
            hc_num_seqs = num_packed_seqs if num_packed_seqs > 1 else None
            if attention_mask is not None:
                valid_tok = attention_mask[:, :T].bool()
                real_pos = torch.where(
                    valid_tok,
                    valid_tok.long().cumsum(dim=1) - 1,
                    torch.full((B, T), -1, device=device, dtype=torch.long),
                )
                row_seq_lens_after = valid_tok.long().sum(dim=1)
                full_kv_pos = real_pos
            else:
                real_pos = None
                row_seq_lens_after = torch.full((B,), T, device=device, dtype=torch.long)
                full_kv_pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T) if want_cache else None
            cur_q_pos = real_pos
            past_kv_pos = None

        if is_decode:
            max_row_len = int(row_seq_lens_after.max().item()) if B > 0 else (past_seq_len + T_step)
            num_attn_blocks = max(self.max_blocks, (max_row_len + self.block_size - 1) // self.block_size)
        else:
            num_attn_blocks = max((T_step + self.block_size - 1) // self.block_size, self.max_blocks)

        if is_decode:
            saved_routes = cache.get("router_routes", None)
            if saved_routes is not None and len(saved_routes) == self.num_blocks:
                router_outputs = list(saved_routes)
            else:
                router_outputs = [None] * self.num_blocks
                ls = cache.get("last_sel", None)
                la = cache.get("last_adv", None)
                if ls is not None and la is not None:
                    for j in range(self.num_blocks - 1):
                        if isinstance(self.blocks[j], GlobalBlock):
                            router_outputs[j] = (ls, la)
            if self.reeval_interval > 0:
                if T_step == 1:
                    reeval_rows = (row_seq_lens_before > 0) & ((row_seq_lens_before % self.reeval_interval) == 0)
                else:
                    reeval_rows = (row_seq_lens_before > 0) & (
                        ((row_seq_lens_after - 1).clamp(min=0) // self.reeval_interval)
                        > ((row_seq_lens_before - 1).clamp(min=0) // self.reeval_interval)
                    )
            else:
                reeval_rows = torch.zeros(B, dtype=torch.bool, device=device)
            reeval_now = bool(reeval_rows.any().item())
            prefill_routing_spec = None
        else:
            router_outputs = [None] * self.num_blocks
            reeval_rows = None
            reeval_now = False
            kv_pos_for_route = full_kv_pos if full_kv_pos is not None else (
                torch.arange(T_step, device=device).unsqueeze(0).expand(B, T_step) if T_step > 0 else torch.zeros(B, 0, device=device, dtype=torch.long)
            )
            seq_for_route = seq_idx[:, :T_step] if seq_idx is not None else None
            prefill_routing_spec = (self.block_size, num_attn_blocks, kv_pos_for_route, seq_for_route)

        router_per_seq = [None] * self.num_blocks
        track_seq_cache = (is_decode or want_cache) and (seq_idx is not None)
        decode_chunk_mask = am_cur if (is_decode and am_cur is not None) else None
        need_hc_state = is_decode or want_cache

        for i in range(self.num_blocks):
            block = self.blocks[i]
            is_global = isinstance(block, GlobalBlock)
            is_last_global = is_global and (i == self.num_blocks - 1)

            if not is_global:
                sel_adv = None
                sel_adv_seq = None
                for k in range(i - 1, -1, -1):
                    if router_outputs[k] is not None:
                        sel_k, adv_k = router_outputs[k]
                        sel_adv = self._align_route_width(sel_k, adv_k, num_attn_blocks)
                        if router_per_seq[k] is not None:
                            sel_adv_seq = router_per_seq[k]
                        break
                if sel_adv is None:
                    B_cur = streams[0].shape[0]
                    sel_d = torch.ones(B_cur, num_attn_blocks, dtype=torch.bool, device=device)
                    adv_d = torch.zeros_like(sel_d)
                    sel_adv = (sel_d, adv_d)
            else:
                sel_adv = None
                sel_adv_seq = None

            def make_layer_fn(i, block, is_global, is_last_global, sel_adv, sel_adv_seq, router_outputs, router_per_seq):
                def layer_fn(h):
                    nonlocal attn_kvs, conv_caches, thought_caches, mamba_inf, attn_seq_idxs, conv_seq_caches
                    if is_global:
                        need_router = not is_last_global
                        force_c = (not is_decode and need_router) or (is_decode and reeval_now and need_router)
                        past_s = attn_seq_idxs[i] if is_decode else None
                        r_spec = prefill_routing_spec if (not is_decode and force_c) else None
                        if track_seq_cache:
                            h_out, new_kv, new_conv, C, new_conv_seq = block(
                                h, past_kv=attn_kvs[i], conv_cache=conv_caches[i],
                                seq_idx=seq_idx, attention_mask=attention_mask,
                                return_importance=force_c,
                                conv_seq_cache=conv_seq_caches[i],
                                past_seq_idx=past_s,
                                return_seq_cache=True,
                                q_pos=cur_q_pos,
                                past_kv_pos=past_kv_pos,
                                routing_spec=r_spec)
                        else:
                            h_out, new_kv, new_conv, C = block(
                                h, past_kv=attn_kvs[i], conv_cache=conv_caches[i],
                                seq_idx=seq_idx, attention_mask=attention_mask,
                                return_importance=force_c,
                                q_pos=cur_q_pos,
                                past_kv_pos=past_kv_pos,
                                routing_spec=r_spec)
                            new_conv_seq = None
                        if need_router and C is not None:
                            if not is_decode and isinstance(C, tuple):
                                score_a, score_b, valid_blk, C_last = C
                                sel_3d, adv_3d = self.block_router.route_causal_from_scores(
                                    score_a, score_b, valid_blk)
                                router_outputs[i] = (sel_3d, adv_3d)
                                if want_cache:
                                    if seq_idx is not None and num_packed_seqs > 1:
                                        kv_pos_eff = full_kv_pos if full_kv_pos is not None else torch.arange(C_last.size(1), device=device).unsqueeze(0).expand(B, C_last.size(1))
                                        C_log, am_log, seq_log = self._compact_importance_and_mask(
                                            C_last, kv_pos_eff, num_attn_blocks,
                                            seq_vals=seq_idx[:, :C_last.size(1)],
                                            target_seq=active_seq_ids)
                                    elif full_kv_pos is not None:
                                        C_log, am_log = self._compact_importance_and_mask(
                                            C_last, full_kv_pos, num_attn_blocks)
                                    else:
                                        C_log = C_last
                                        am_log = attention_mask
                                    sel_2d, adv_2d = self.block_router(
                                        C_log, self.block_size, num_attn_blocks,
                                        attention_mask=am_log)
                                    router_per_seq[i] = (sel_2d, adv_2d)
                            elif not is_decode and seq_idx is not None and num_packed_seqs > 1:
                                kv_pos_eff = full_kv_pos if full_kv_pos is not None else torch.arange(C.size(1), device=device).unsqueeze(0).expand(B, C.size(1))
                                C_log, am_log, seq_log = self._compact_importance_and_mask(
                                    C, kv_pos_eff, num_attn_blocks, seq_vals=seq_idx[:, :C.size(1)])
                                sel_s, adv_s = self.block_router.route_packed(
                                    C_log, self.block_size, num_attn_blocks, seq_log,
                                    num_packed_seqs, attention_mask=am_log)
                                router_per_seq[i] = (sel_s, adv_s)
                                b_all_r = torch.arange(B, device=device)
                                active_idx = (active_seq_ids - 1).clamp(min=0, max=num_packed_seqs - 1)
                                has_act = active_seq_ids >= 1
                                sel = torch.where(has_act.unsqueeze(1), sel_s[b_all_r, active_idx], torch.zeros(B, num_attn_blocks, dtype=torch.bool, device=device))
                                adv = torch.where(has_act.unsqueeze(1), adv_s[b_all_r, active_idx], torch.zeros(B, num_attn_blocks, dtype=torch.bool, device=device))
                                router_outputs[i] = (sel, adv)
                            elif full_kv_pos is not None:
                                full_s_vals = torch.cat([past_s, seq_idx[:, :T_step]], dim=1) if (is_decode and past_s is not None and seq_idx is not None) else (seq_idx[:, :C.size(1)] if seq_idx is not None else None)
                                if full_s_vals is not None and active_seq_ids is not None:
                                    C_log, am_log, _ = self._compact_importance_and_mask(
                                        C, full_kv_pos, num_attn_blocks, seq_vals=full_s_vals, target_seq=active_seq_ids)
                                else:
                                    C_log, am_log = self._compact_importance_and_mask(C, full_kv_pos, num_attn_blocks)
                                new_sel, new_adv = self.block_router(
                                    C_log, self.block_size, num_attn_blocks,
                                    attention_mask=am_log)
                                if is_decode and router_outputs[i] is not None:
                                    old_sel, old_adv = self._align_route_width(
                                        router_outputs[i][0], router_outputs[i][1], num_attn_blocks)
                                    r_col = reeval_rows.unsqueeze(1)
                                    new_sel = torch.where(r_col, new_sel, old_sel)
                                    new_adv = torch.where(r_col, new_adv, old_adv)
                                router_outputs[i] = (new_sel, new_adv)
                            else:
                                new_sel, new_adv = self.block_router(
                                    C, self.block_size, num_attn_blocks,
                                    attention_mask=attention_mask)
                                if is_decode and router_outputs[i] is not None:
                                    old_sel, old_adv = self._align_route_width(
                                        router_outputs[i][0], router_outputs[i][1], num_attn_blocks)
                                    r_col = reeval_rows.unsqueeze(1)
                                    new_sel = torch.where(r_col, new_sel, old_sel)
                                    new_adv = torch.where(r_col, new_adv, old_adv)
                                router_outputs[i] = (new_sel, new_adv)
                        if is_decode or want_cache:
                            attn_kvs[i] = new_kv
                            conv_caches[i] = new_conv
                            if track_seq_cache:
                                conv_seq_caches[i] = new_conv_seq
                                if is_decode and past_s is not None:
                                    attn_seq_idxs[i] = torch.cat([past_s, seq_idx[:, :T_step]], dim=1)
                                elif not is_decode:
                                    attn_seq_idxs[i] = seq_idx[:, :T_step]
                        return h_out
                    else:
                        sel, adv = sel_adv
                        sel_ps, adv_ps = (None, None)
                        if track_seq_cache:
                            h_out, new_attn_cache, new_conv, new_conv_seq = block(
                                h, cache=thought_caches[i], conv_cache=conv_caches[i],
                                seq_idx=seq_idx,
                                attention_mask=attention_mask,
                                selected=sel, advanced=adv,
                                reeval=reeval_now,
                                build_cache=(want_cache and not is_decode),
                                mamba_inference_params=mamba_inf,
                                conv_seq_cache=conv_seq_caches[i],
                                cur_seq_ids=active_seq_ids,
                                return_seq_cache=True,
                                real_pos=real_pos,
                                selected_per_seq=sel_ps,
                                advanced_per_seq=adv_ps,
                                reeval_mask=reeval_rows,
                                chunk_mask=decode_chunk_mask)
                        else:
                            h_out, new_attn_cache, new_conv = block(
                                h, cache=thought_caches[i], conv_cache=conv_caches[i],
                                seq_idx=seq_idx,
                                attention_mask=attention_mask,
                                selected=sel, advanced=adv,
                                reeval=reeval_now,
                                build_cache=(want_cache and not is_decode),
                                mamba_inference_params=mamba_inf,
                                real_pos=real_pos,
                                selected_per_seq=sel_ps,
                                advanced_per_seq=adv_ps,
                                reeval_mask=reeval_rows,
                                chunk_mask=decode_chunk_mask)
                            new_conv_seq = None
                        if is_decode or want_cache:
                            conv_caches[i] = new_conv
                            if track_seq_cache:
                                conv_seq_caches[i] = new_conv_seq
                            if new_attn_cache is not None:
                                thought_caches[i] = new_attn_cache
                        return h_out
                return layer_fn

            layer_fn = make_layer_fn(i, block, is_global, is_last_global, sel_adv, sel_adv_seq, router_outputs, router_per_seq)

            if self.use_checkpoint and self.training and torch.is_grad_enabled() and not is_decode:
                def wrapped(h, _layer_fn=layer_fn):
                    return checkpoint(_layer_fn, h, use_reentrant=True)
            else:
                wrapped = layer_fn

            hc_in_state = hc_router_states[i] if is_decode else None
            hc_res = self.hcs[i](
                streams, wrapped,
                attention_mask=hc_mask,
                seq_idx=hc_seq_idx if not is_decode else None,
                num_seqs=hc_num_seqs if not is_decode else None,
                router_state=hc_in_state,
                return_router_state=need_hc_state,
            )
            if need_hc_state:
                streams, hc_router_states[i] = hc_res
            else:
                streams = hc_res

        w = F.softmax(self.final_mix, dim=0)
        x = sum(w[j] * streams[j] for j in range(self.num_streams))
        x = self.norm(x)

        if mamba_inf is not None:
            mamba_inf.seqlen_offset += T_step

        new_cache = None
        if want_cache:
            new_seq_len = (past_seq_len + T_step) if is_decode else T_step
            if not is_decode:
                cache_routes = [None] * self.num_blocks
                for j in range(self.num_blocks):
                    if router_per_seq[j] is not None:
                        cache_routes[j] = router_per_seq[j]
                    elif router_outputs[j] is not None:
                        sel_j, adv_j = router_outputs[j]
                        if sel_j.dim() == 3:
                            if sel_j.size(1) > 0:
                                cache_routes[j] = (sel_j[:, -1, :], adv_j[:, -1, :])
                        else:
                            cache_routes[j] = (sel_j, adv_j)
            else:
                cache_routes = router_outputs
            last_sel = cache.get("last_sel", None) if is_decode else None
            last_adv = cache.get("last_adv", None) if is_decode else None
            for j in range(self.num_blocks - 1, -1, -1):
                if cache_routes[j] is not None:
                    last_sel, last_adv = cache_routes[j]
                    break
            new_cache = {
                "attn_kvs": attn_kvs,
                "conv_caches": conv_caches,
                "thought_caches": thought_caches,
                "mamba_inference_params": mamba_inf,
                "router_routes": cache_routes,
                "last_sel": last_sel,
                "last_adv": last_adv,
                "last_tokens": new_last_tokens,
                "seq_len": new_seq_len,
                "row_seq_lens": row_seq_lens_after,
                "attn_kv_pos": full_kv_pos,
                "active_seq_ids": active_seq_ids,
                "attn_seq_idxs": attn_seq_idxs,
                "conv_seq_caches": conv_seq_caches,
                "hc_router_states": hc_router_states,
            }
        return x, new_cache


class CouncilModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = CouncilBackbone(
            d_model=config.d_model_backbone,
            d_model_thought=config.d_model_thought,
            num_blocks=config.num_blocks,
            num_streams=config.num_streams,
            vocab_size=config.vocab_size,
            use_checkpoint=getattr(config, "backbone_checkpoint", False),
            window_size=getattr(config, "sliding_window", 4096),
            attention_checkpoint=getattr(config, "attention_checkpoint", None),
            config=config)
        self.backbone.config_mamba_headdim = config.mamba_headdim
        self.final_norm = nn.LayerNorm(config.d_model_backbone)
        self.lm_head = nn.Linear(config.d_model_backbone, config.vocab_size, bias=False)
        self.lm_head.weight = self.backbone.embedding.weight
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.backbone.embedding.weight, mean=0, std=0.02)

    def forward(self, input_ids, labels=None, num_recursion=None, cache=None,
                return_cache=False, force_full_recursion=False, cu_seqlens=None,
                attention_mask=None, return_loss_info=False):
        x, new_cache = self.backbone(input_ids, cache=cache, return_cache=return_cache,
                                       cu_seqlens=cu_seqlens, attention_mask=attention_mask)
        x_out = self.final_norm(x)
        logits = self.lm_head(x_out)
        if labels is None:
            if return_cache:
                return logits, new_cache
            return logits
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:].clone()
        if cu_seqlens is not None:
            if cu_seqlens.dim() == 1:
                cu_seqlens = cu_seqlens.unsqueeze(0)
            B_labels = labels.size(0)
            B_cu = cu_seqlens.size(0)
            if B_cu == 1 and B_labels > 1:
                cu_eff = cu_seqlens.expand(B_labels, -1)
            elif B_cu == B_labels:
                cu_eff = cu_seqlens
            else:
                raise ValueError(f"cu_seqlens batch size {B_cu} does not match labels batch size {B_labels}")
            for b in range(B_labels):
                row = cu_eff[b].to(torch.long)
                valid_vals = row[row > 0]
                if valid_vals.numel() > 0:
                    max_end = int(valid_vals.max().item())
                    if max_end < labels.size(1):
                        shift_labels[b, max(max_end - 1, 0):] = -100
                elif (row >= 0).any() and row.numel() >= 2:
                    shift_labels[b, :] = -100
                pos = row[1:] - 1
                valid = (pos >= 0) & (pos < shift_labels.size(1))
                pos = pos[valid]
                shift_labels[b, pos] = -100
        if attention_mask is not None and attention_mask.size(1) == labels.size(1):
            shift_am_src = attention_mask[..., :-1]
            shift_am_tgt = attention_mask[..., 1:]
            shift_labels[shift_am_src == 0] = -100
            shift_labels[shift_am_tgt == 0] = -100
        B_, T_, V_ = shift_logits.shape
        chunk_size = 1024
        total_loss = torch.zeros((), device=shift_logits.device, dtype=torch.float32)
        total_valid = torch.zeros((), device=shift_logits.device, dtype=torch.float32)
        flat_logits = shift_logits.reshape(-1, V_)
        flat_labels = shift_labels.reshape(-1)
        for i in range(0, B_ * T_, chunk_size):
            cl = flat_logits[i:i + chunk_size].float()
            cla = flat_labels[i:i + chunk_size]
            chunk_loss = F.cross_entropy(cl, cla, ignore_index=-100, reduction='sum')
            total_loss = total_loss + chunk_loss
            total_valid = total_valid + (cla != -100).sum()
        loss = total_loss / total_valid.clamp(min=1)
        if return_loss_info:
            if return_cache:
                return logits, loss, total_valid, new_cache
            return logits, loss, total_valid
        if return_cache:
            return logits, loss, new_cache
        return logits, loss

    @torch.no_grad()
    def probe_recursion(self, input_ids, labels, max_rec=None, delta_loss_threshold=0.05):
        return None

    @staticmethod
    def _apply_repetition_penalty(logits, generated_ids, penalty, attention_mask=None):
        if penalty is None or penalty <= 1.0:
            return logits
        if attention_mask is None:
            score = torch.gather(logits, 1, generated_ids)
            score = torch.where(score < 0, score * penalty, score / penalty)
            return logits.scatter(1, generated_ids, score)
        logits = logits.clone()
        B = generated_ids.size(0)
        am = attention_mask.bool()
        for b in range(B):
            ids_b = generated_ids[b][am[b]]
            if ids_b.numel() == 0:
                continue
            ids_b = torch.unique(ids_b)
            score = logits[b, ids_b]
            score = torch.where(score < 0, score * penalty, score / penalty)
            logits[b, ids_b] = score
        return logits

    @staticmethod
    def _apply_no_repeat_ngram(logits, generated_ids, n, attention_mask=None):
        if n is None or n <= 0:
            return logits
        B = generated_ids.size(0)
        logits = logits.clone()
        am = attention_mask.bool() if attention_mask is not None else None
        for b in range(B):
            if am is not None:
                seq = generated_ids[b][am[b]].tolist()
            else:
                seq = generated_ids[b].tolist()
            if len(seq) < n:
                continue
            prefix = tuple(seq[-(n - 1):]) if n > 1 else ()
            banned = set()
            for i in range(len(seq) - n + 1):
                ng = tuple(seq[i:i + n])
                if ng[:n - 1] == prefix:
                    banned.add(ng[-1])
            if banned:
                idx = torch.tensor(sorted(banned), device=logits.device, dtype=torch.long)
                logits[b, idx] = float("-inf")
        return logits

    def _sample_next(self, nxt_logits, temperature):
        if temperature is None or temperature <= 0:
            return nxt_logits.argmax(dim=-1, keepdim=True)
        nxt_logits = nxt_logits.float() / max(temperature, 1e-6)
        probs = F.softmax(nxt_logits, dim=-1)
        probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
        row_sum = probs.sum(dim=-1, keepdim=True)
        uniform = torch.full_like(probs, 1.0 / probs.shape[-1])
        probs = torch.where(row_sum > 0, probs / row_sum.clamp(min=1e-12), uniform)
        return torch.multinomial(probs, num_samples=1)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=512,
                 temperature=0.8, eos_token_id=None, repetition_penalty=1.15,
                 no_repeat_ngram_size=0, cu_seqlens=None):
        was_training = self.training
        self.eval()
        try:
            B = input_ids.size(0)
            T_p = input_ids.size(1)
            device = input_ids.device
            if T_p == 0:
                raise ValueError("input_ids must have non-zero sequence length")
            if attention_mask is None:
                attention_mask = torch.ones(B, T_p, device=device, dtype=torch.long)
            else:
                attention_mask = attention_mask.long()
            am_bool = attention_mask[:, :T_p].bool()
            if cu_seqlens is not None:
                seq_idx_p = self.backbone._build_seq_idx_from_cu_seqlens(
                    cu_seqlens, T_p, attention_mask, device, target_batch=B)
                active_seq_p = self.backbone._extract_active_seq_ids(seq_idx_p, T_p)
                token_mask_p = am_bool & (seq_idx_p == active_seq_p.view(B, 1)) & (active_seq_p.view(B, 1) >= 0)
            else:
                token_mask_p = am_bool
            if not token_mask_p.any(dim=1).all().item():
                raise ValueError("Each prompt in generate() must contain at least one non-padding token")
            if max_new_tokens <= 0:
                return input_ids

            k = self.backbone.engram.max_ngram - 1
            cfg_best_N = getattr(self.config, "best_N", 0)
            n_rec = cfg_best_N if cfg_best_N > 0 else self.config.rl_generate_recursion
            force_full = not getattr(self.config, "adaptive_inference", True)

            logits, cache = self.forward(input_ids, attention_mask=attention_mask,
                                           num_recursion=n_rec, force_full_recursion=force_full,
                                           cu_seqlens=cu_seqlens, return_cache=True)

            pos_grid = torch.arange(T_p, device=device).unsqueeze(0).expand(B, T_p)
            masked_pos = torch.where(token_mask_p, pos_grid, torch.full_like(pos_grid, -1))
            last_pos = masked_pos.max(dim=1).values
            if k > 0:
                last_tokens = torch.full((B, k), -1, device=device, dtype=input_ids.dtype)
                lengths = token_mask_p.long().sum(dim=1)
                rank = token_mask_p.long().cumsum(dim=1)
                valid_src = token_mask_p & (rank > (lengths.unsqueeze(1) - k))
                if valid_src.any():
                    b_idx, t_idx = valid_src.nonzero(as_tuple=True)
                    dst_idx = rank[b_idx, t_idx] - lengths[b_idx] + (k - 1)
                    last_tokens[b_idx, dst_idx] = input_ids[b_idx, t_idx]
            else:
                last_tokens = input_ids.new_zeros(B, 0)
            out = input_ids
            penalty_mask = token_mask_p.long()
            finished = torch.zeros(B, dtype=torch.bool, device=device)
            nxt_logits = logits[torch.arange(B, device=device), last_pos, :]
            nxt_logits = self._apply_repetition_penalty(nxt_logits, out, repetition_penalty, attention_mask=penalty_mask)
            nxt_logits = self._apply_no_repeat_ngram(nxt_logits, out, no_repeat_ngram_size, attention_mask=penalty_mask)
            nxt = self._sample_next(nxt_logits, temperature)
            if eos_token_id is not None:
                nxt = torch.where(finished.unsqueeze(-1), torch.full_like(nxt, eos_token_id), nxt)
            out = torch.cat([out, nxt], dim=-1)
            if eos_token_id is not None:
                finished = finished | (nxt.squeeze(-1) == eos_token_id)
                if finished.all().item():
                    return out
            for _ in range(max_new_tokens - 1):
                cache["last_tokens"] = last_tokens
                attention_mask = torch.cat(
                    [attention_mask, torch.ones(B, 1, device=device, dtype=attention_mask.dtype)],
                    dim=1)
                penalty_mask = torch.cat(
                    [penalty_mask, torch.ones(B, 1, device=device, dtype=penalty_mask.dtype)],
                    dim=1)
                logits, cache = self.forward(nxt, attention_mask=attention_mask,
                                               num_recursion=n_rec, cache=cache,
                                               force_full_recursion=force_full, return_cache=True)
                if k > 0:
                    last_tokens = torch.cat([last_tokens, nxt], dim=1)[:, -k:]
                else:
                    last_tokens = input_ids.new_zeros(B, 0)
                nxt_logits = logits[:, -1, :]
                nxt_logits = self._apply_repetition_penalty(nxt_logits, out, repetition_penalty, attention_mask=penalty_mask)
                nxt_logits = self._apply_no_repeat_ngram(nxt_logits, out, no_repeat_ngram_size, attention_mask=penalty_mask)
                nxt = self._sample_next(nxt_logits, temperature)
                if eos_token_id is not None:
                    nxt = torch.where(finished.unsqueeze(-1), torch.full_like(nxt, eos_token_id), nxt)
                out = torch.cat([out, nxt], dim=-1)
                if eos_token_id is not None:
                    finished = finished | (nxt.squeeze(-1) == eos_token_id)
                    if finished.all().item():
                        break
            return out
        finally:
            self.train(was_training)
