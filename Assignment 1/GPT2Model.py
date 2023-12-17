import torch
import torch.nn.functional as F
from dataclasses import dataclass

class MultiheadAttention():
    def __init__(self, embed_dim:int=768, n_heads:int=12, bias:bool=True, dropout:int=0.1) -> None:
        assert embed_dim % n_heads == 0
        
        self.c_attn = torch.nn.Linear(embed_dim, 3*embed_dim, bias=bias)
        self.c_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_dropout = torch.nn.Dropout(dropout)
        self.resid_dropout = torch.nn.Dropout(dropout)
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.dropout = dropout

    def forward(self, x:torch.Tensor):
        B,T,C = x.size()
        q,k,v = self.c_attn(x).split(self.embed_dim, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = F.scaled_dot_product_attention(q,k,v,None,self.dropout,True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class DecoderBlock(torch.nn.Module):
    def __init__(self, embed_dim, n_heads) -> None:
        super(DecoderBlock, self).__init__()
        self.ln_1 = torch.nn.LayerNorm(embed_dim)
        self.attn = MultiheadAttention(embed_dim, n_heads, True, 0.1)
        self.ln_2 = torch.nn.LayerNorm(embed_dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, 4*embed_dim),
            torch.nn.GELU(),
            torch.nn.Linear(4*embed_dim, embed_dim)
        )
    
    def forward(self,x):
        a, _ = self.attn(self.ln_1(x))
        x = x + a
        x = x + self.mlp(self.ln_2(x))
        return x

class DecoderBlockV2(torch.nn.Module):
    def __init__(self, embed_dim, n_heads) -> None:
        self.ln_1 = torch.nn.LayerNorm(embed_dim)
        self.attn = torch.nn.MultiheadAttention(embed_dim, n_heads, 0.1, batch_first=True)
        self.ln_2 = torch.nn.LayerNorm(embed_dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, 4*embed_dim),
            torch.nn.GELU(),
            torch.nn.Linear(4*embed_dim, embed_dim)
        )
    
    def forward(self,x):
        a, _ = self.attn(self.ln_1(x))
        x = x + a
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2(torch.nn.Module):
    def __init__(self, embed_dim:int=768, n_layers:int=12, block_size:int=1024, vocab_size:int=50257, n_heads:int=12) -> None:
        super().__init__()
        self.block_size = block_size
        self.wte = torch.nn.Embedding(vocab_size, embed_dim)
        self.wpe = torch.nn.Embedding(block_size, embed_dim)
        self.drop = torch.nn.Dropout(0.1)
        self.h = torch.nn.ModuleList([DecoderBlock(embed_dim, n_heads) for _ in range(n_layers)])
        self.ln_f = torch.nn.LayerNorm(embed_dim)
        self.lm_head = torch.nn.Linear(embed_dim, vocab_size, bias=False)

        self.wte.weight = self.lm_head.weight
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
        token_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(token_emb+pos_emb)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)

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
       
        model = GPT2()
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] 

        model_hf = GPT2LMHeadModel.from_pretrained('gpt2')
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] 
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] 
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model