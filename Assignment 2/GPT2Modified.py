import torch
import torch.nn.functional as F
import math
from typing import Optional
from einops import rearrange, einsum
from dataclasses import dataclass

def scaled_dot_product_gqa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout: float = 0.0,
    scale: Optional[float] = None,
    mask: Optional[torch.Tensor] = None,
    is_causal: Optional[bool] = None,
    need_weights: bool = False,
    average_attn_weights: bool = False,
    force_grouped: bool = False,
):
    """Scaled dot product attention with support for grouped queries.

    Einstein notation:
    - b: batch size
    - n / s: sequence length
    - h: number of heads
    - g: number of groups
    - d: dimension of query/key/value

    Args:
        query: Query tensor of shape (b, n, h, d)
        key: Key tensor of shape (b, s, h, d)
        value: Value tensor of shape (b, s, h, d)
        dropout: Dropout probability (default: 0.0)
        scale: Scale factor for query (default: d_query ** 0.5)
        mask: Mask tensor of shape (b, n, s) or (b, s). If 'ndim == 2', the mask is
            applied to all 'n' rows of the attention matrix. (default: None)
        force_grouped: If True, apply grouped-query attention even if the number of
            heads is equal for query, key, and value. (default: False)

    Returns:
        2-tuple of:
        - Attention output with shape (b, n, h, d)
        - (Optional) Attention weights with shape (b, h, n, s). Only returned if
          'need_weights' is True.
    """
    if (mask is not None) and (is_causal is not None):
        raise ValueError(
            "Only one of 'mask' and 'is_causal' should be provided, but got both."
        )
    elif not query.ndim == key.ndim == value.ndim == 4:
        raise ValueError(
            f"Expected query, key, and value to be 4-dimensional, but got shapes "
            f"{query.shape}, {key.shape}, and {value.shape}."
        )

    # Move sequence length dimension to axis 2.
    # This makes the attention operations below *much* faster.
    query = rearrange(query, "b n h d -> b h n d")
    key = rearrange(key, "b s h d -> b h s d")
    value = rearrange(value, "b s h d -> b h s d")

    bq, hq, nq, dq = query.shape
    bk, hk, nk, dk = key.shape
    bv, hv, nv, dv = value.shape
    if not (bq == bk == bv and dq == dk == dv):
        raise ValueError(
            "Expected query, key, and value to have the same batch size (dim=0) and "
            f"embedding dimension (dim=3), but got query: {query.shape}, "
            f"key: {key.shape}, and value: {value.shape}."
        )
    elif (hk != hv) or (nk != nv):
        raise ValueError(
            "Expected key and value to have the same size in dimensions 1 and 2, but "
            f"got key: {key.shape} and value: {value.shape}."
        )
    elif hq % hk != 0:
        raise ValueError(
            "Expected query heads to be a multiple of key/value heads, but got "
            f"query: {query.shape} and key/value: {key.shape}."
        )

    if scale is None:
        scale = query.size(-1) ** 0.5
    query = query / scale

    num_head_groups = hq // hk
    if num_head_groups > 1 or force_grouped:
        # Separate the query heads into 'num_head_groups' chunks, and fold the group
        # dimension into the batch dimension.  This allows us to compute the attention
        # for each head in parallel, then sum over all of the groups at the end.
        query = rearrange(query, "b (h g) n d -> b g h n d", g=num_head_groups)
        similarity = einsum(query, key, "b g h n d, b h s d -> b h n s")
    else:
        # If the number of query/key heads is equal, we can skip grouping the queries,
        # and just use the standard sdot product attention.
        similarity = einsum(query, key, "b h n d, b h s d -> b h n s")

    if is_causal:
        # Mask out the upper triangular portion of the attention matrix. This prevents
        # the model from attending to tokens in the future.
        mask = torch.ones(
            (bq, nq, nk),
            device=query.device,
            dtype=torch.bool,
        ).tril_()

    if mask is not None:
        # Expand mask to match the shape of the attention matrix.
        # If mask is 2D, assume that it is applied to the key/value sequence dimension.
        # Else if mask is 3D, assume that it is applied to the query/key/value sequence
        # dimension for all attention heads.
        #
        # Users could also provide a 4D mask, which is applied to the query/key/value
        # sequence dimension for each attention head (though I don't have a particular
        # use case in mind for that).
        if mask.ndim == 2:
            mask = rearrange(mask, "b s -> b () () s")
        elif mask.ndim == 3:
            mask = rearrange(mask, "b n s -> b () n s")
        # Mask similarity values by setting them to negative infinity.  This guarantees
        # that they will not contribute to the softmax computation below.
        similarity.masked_fill_(~mask, torch.finfo(similarity.dtype).min)

    attention = F.softmax(similarity / scale, dim=-1)
    if dropout > 0.0:
        attention = F.dropout(attention, p=dropout)

    # Apply attention matrix to the value Tensor.
    out = einsum(attention, value, "b h n s, b h s d -> b h n d")
    # Move head dimension back to axis 2
    out = rearrange(out, "b h n d -> b n h d")

    attn_weights: Optional[torch.Tensor] = None
    if need_weights:
        # Move the sequence dimensions back to positions 1, 2.  Move the head dimension
        # to position 3.  This more closely matches the return shape of the attention
        # output: (b, n, h, d).
        attn_weights = rearrange(attention, "b h n s -> b n s h")
        if average_attn_weights:
            attn_weights = attn_weights.mean(dim=1)

    return out, attn_weights

class MultiheadAttention(torch.nn.Module):
    def __init__(self, embed_dim:int=768, n_heads:int=12, bias:bool=True, dropout=0.1):
        super().__init__()
        assert embed_dim % n_heads == 0

        self.c_attn = torch.nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.c_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_dropout = torch.nn.Dropout(dropout)
        self.resid_dropout = torch.nn.Dropout(dropout)
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.dropout = dropout
  
    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.embed_dim, dim=2)

        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) 
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) 

        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y
    
class MLP(torch.nn.Module):
    def __init__(self, embed_dim:int=768, bias:bool=True, dropout=0.1) -> None:
        super(MLP, self).__init__()
        self.c_fc = torch.nn.Linear(embed_dim, 4*embed_dim, bias=bias)
        self.gelu = torch.nn.GELU()
        self.c_proj = torch.nn.Linear(4*embed_dim, embed_dim, bias=bias)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class DecoderBlock(torch.nn.Module):
    def __init__(self, embed_dim, n_heads) -> None:
        super(DecoderBlock, self).__init__()
        self.ln_1 = torch.nn.LayerNorm(embed_dim)
        self.attn = MultiheadAttention(embed_dim, n_heads, True, 0.1)
        self.ln_2 = torch.nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim)
    
    def forward(self,x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2(torch.nn.Module):
    def __init__(self, embed_dim:int=768, n_layers:int=12, block_size:int=1024, vocab_size:int=50257, n_heads:int=12) -> None:
        super().__init__()
        self.block_size = block_size
        self.transformer = torch.nn.ModuleDict(dict(
            wte = torch.nn.Embedding(vocab_size, embed_dim),
            wpe = torch.nn.Embedding(block_size, embed_dim),
            drop = torch.nn.Dropout(0.1),
            h = torch.nn.ModuleList([DecoderBlock(embed_dim, n_heads) for _ in range(n_layers)]),
            ln_f = torch.nn.LayerNorm(embed_dim)
        ))
        self.lm_head = torch.nn.Linear(embed_dim, vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        token_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(token_emb+pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    
    @classmethod
    def from_pretrained(cls, override_args=None):
        override_args = override_args or {}
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
       
        coded_model = GPT2()
        sd = coded_model.state_dict()
        coded_model_keys = sd.keys()
        coded_model_keys = [k for k in coded_model_keys if not k.endswith('.attn.bias')] 

        loaded_model = GPT2LMHeadModel.from_pretrained('gpt2')
        loaded_sd = loaded_model.state_dict()

        loaded_model_keys = loaded_sd.keys()
        loaded_model_keys = [k for k in loaded_model_keys if not k.endswith('.attn.masked_bias')] 
        loaded_model_keys = [k for k in loaded_model_keys if not k.endswith('.attn.bias')] 
        
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        for k in loaded_model_keys:
            if any(k.endswith(w) for w in transposed):
                assert loaded_sd[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(loaded_sd[k].t())
            else:
                assert loaded_sd[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(loaded_sd[k])

        return coded_model