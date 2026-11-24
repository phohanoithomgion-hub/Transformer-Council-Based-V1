import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

try:
    _COMPILED_FLEX_ATTENTION = torch.compile(
        flex_attention,
        dynamic=True,
        fullgraph=True,
        mode="default",
    )
except Exception:
    _COMPILED_FLEX_ATTENTION = flex_attention


class MLAAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_c=512, d_rotate=64, window_size=4096):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_c = d_c
        self.d_rotate = d_rotate
        self.window_size = window_size

        self.q_down = nn.Linear(d_model, d_c, bias=False)
        self.kv_down = nn.Linear(d_model, d_c, bias=False)
        self.q_up = nn.Linear(d_c, d_model, bias=False)
        self.k_up = nn.Linear(d_c, d_model, bias=False)
        self.v_up = nn.Linear(d_c, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def _compute_positions_from_seq(self, seq_idx, B, T, device):
        is_new = torch.zeros(B, T, dtype=torch.bool, device=device)
        is_new[:, 0] = True
        if T > 1:
            is_new[:, 1:] = seq_idx[:, 1:] != seq_idx[:, :-1]
        seg_start = torch.where(
            is_new,
            torch.arange(T, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1),
            torch.zeros(1, T, dtype=torch.long, device=device),
        )
        seg_start = torch.cummax(seg_start, dim=1)[0]
        return torch.arange(T, device=device, dtype=torch.long).unsqueeze(0) - seg_start

    def _apply_rope(self, x, positions, rot_dim):
        x_rot = x[..., -rot_dim:]
        x_pass = x[..., :-rot_dim]
        half = rot_dim // 2
        freqs = 1.0 / (10000.0 ** (torch.arange(0, half, device=x.device).float() / half))
        angles = positions.float().unsqueeze(-1) * freqs
        cos_a = angles.cos()
        sin_a = angles.sin()
        x_rot_pair = x_rot.reshape(*x_rot.shape[:-1], half, 2)
        x0 = x_rot_pair[..., 0]
        x1 = x_rot_pair[..., 1]
        if positions.dim() == 1:
            x0_new = x0 * cos_a - x1 * sin_a
            x1_new = x0 * sin_a + x1 * cos_a
        else:
            cos_b = cos_a.unsqueeze(1)
            sin_b = sin_a.unsqueeze(1)
            x0_new = x0 * cos_b - x1 * sin_b
            x1_new = x0 * sin_b + x1 * cos_b
        x_rot_new = torch.stack([x0_new, x1_new], dim=-1).reshape(*x_rot.shape)
        return torch.cat([x_pass, x_rot_new.to(x.dtype)], dim=-1)

    def forward(self, x, past_kv=None, seq_idx=None, attention_mask=None):
        B, T, D = x.shape
        q_c = self.q_down(x)
        kv_c = self.kv_down(x)

        q = self.q_up(q_c).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_up(kv_c).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_up(kv_c).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        rot_dim = self.d_rotate // self.num_heads
        rot_dim = (rot_dim // 2) * 2
        if rot_dim > 0:
            if attention_mask is not None:
                pos_all = attention_mask.long().cumsum(dim=1) - 1
                if past_kv is not None:
                    positions = pos_all[:, -T:]
                else:
                    positions = pos_all[:, :T]
            elif past_kv is not None:
                past_T = past_kv[0].size(2)
                positions = torch.arange(past_T, past_T + T, device=x.device)
            elif seq_idx is not None:
                positions = self._compute_positions_from_seq(seq_idx, B, T, x.device)
            else:
                positions = torch.arange(T, device=x.device)
            q = self._apply_rope(q, positions, rot_dim)
            k = self._apply_rope(k, positions, rot_dim)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv = (k, v)
        T_q = q.size(2)
        T_kv = k.size(2)
        needs_mask = (attention_mask is not None) or (seq_idx is not None)

        if past_kv is not None:
            if T_q == 1:
                attn_mask = None
                if attention_mask is not None:
                    am = attention_mask[:, :T_kv].bool()
                    if self.window_size > 0:
                        cum = am.long().cumsum(dim=1)
                        total = cum[:, -1:]
                        keep_from = (total - self.window_size).clamp(min=0)
                        wm = am & (cum > keep_from)
                        attn_mask = wm[:, None, None, :]
                    else:
                        attn_mask = am[:, None, None, :]
                elif T_kv > self.window_size:
                    k_pos = torch.arange(T_kv, device=x.device)
                    wm = (k_pos >= (T_kv - self.window_size))
                    attn_mask = wm[None, None, None, :]
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
            else:
                q_pos = torch.arange(T_kv - T_q, T_kv, device=x.device).unsqueeze(1)
                k_pos = torch.arange(T_kv, device=x.device).unsqueeze(0)
                m = (k_pos <= q_pos) & (k_pos >= q_pos - self.window_size + 1)
                if attention_mask is not None:
                    m = m.unsqueeze(0) & attention_mask[:, :T_kv].bool().unsqueeze(1)
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=m, is_causal=False)
        elif not needs_mask and T_q <= self.window_size:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            try:
                am = attention_mask
                si = seq_idx
                def mask_mod(b, h, q_idx, kv_idx):
                    diag = q_idx == kv_idx
                    base = (kv_idx <= q_idx) & (kv_idx >= (q_idx - self.window_size + 1))
                    if si is not None:
                        base = base & (si[b, q_idx] == si[b, kv_idx])
                    if am is not None:
                        base = base & am[b, kv_idx].bool()
                    return base | diag
                B_mask = si.size(0) if si is not None else None
                block_mask = create_block_mask(
                    mask_mod, B=B_mask, H=None, Q_LEN=T_q, KV_LEN=T_kv,
                    device=x.device, BLOCK_SIZE=128,
                )
                out = _COMPILED_FLEX_ATTENTION(q, k, v, block_mask=block_mask)
            except Exception:
                q_pos = torch.arange(T_kv - T_q, T_kv, device=x.device).unsqueeze(1)
                k_pos = torch.arange(T_kv, device=x.device).unsqueeze(0)
                mask = (k_pos <= q_pos) & (k_pos >= q_pos - self.window_size + 1)
                if seq_idx is not None:
                    same = (seq_idx.unsqueeze(1) == seq_idx.unsqueeze(2))[:, -T_q:, :]
                    mask = mask.unsqueeze(0) & same
                if attention_mask is not None:
                    if mask.dim() == 2:
                        mask = mask.unsqueeze(0)
                    mask = mask & attention_mask[:, :T_kv].bool().unsqueeze(-2)
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)

        out = out.transpose(1, 2).contiguous().view(B, T_q, D)
        return self.out_proj(out), new_kv