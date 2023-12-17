import torch
import torch.nn.functional as F
import math
from dataclasses import dataclass

class MultiheadAttention(torch.nn.Module):
    """
    Multihead Attention module.

    Args:
        embed_dim (int, optional): Dimension of input embeddings. Default is 768.
        n_heads (int, optional): Number of attention heads. Default is 12.
        bias (bool, optional): If True, learnable bias is added to the linear transformations. Default is True.
        dropout (float, optional): Dropout probability. Default is 0.1.

    Attributes:
        c_attn (torch.nn.Linear): Linear layer for computing attention weights.
        c_proj (torch.nn.Linear): Linear layer for projecting attention output.
        attn_dropout (torch.nn.Dropout): Dropout layer for attention weights.
        resid_dropout (torch.nn.Dropout): Dropout layer for residual connections.
        n_heads (int): Number of attention heads.
        embed_dim (int): Dimension of input embeddings.
        dropout (float): Dropout probability.

    Note:
        The input tensor `x` is expected to have dimensions (B, T, C), where
        B is the batch size, T is the sequence length, and C is the input embedding dimension.
    """

    def __init__(self, embed_dim: int = 768, n_heads: int = 12, bias: bool = True, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % n_heads == 0

        self.c_attn = torch.nn.Linear(embed_dim, 3 * embed_dim, bias=bias)      # Linear layer for computing attention weights
        self.c_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)          # Linear layer for projecting attention output

        # Dropout layers
        self.attn_dropout = torch.nn.Dropout(dropout)
        self.resid_dropout = torch.nn.Dropout(dropout)

        # Number of attention heads, input embedding dimension, and dropout probability
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.dropout = dropout
  
    def forward(self, x):
        """
        Forward pass of the Multihead Attention module.

        Args:
            x (torch.Tensor): Input tensor with dimensions (B, T, C), where B is the batch size, T is the sequence length, and C is the input embedding dimension.

        Returns:
            torch.Tensor: Output tensor with dimensions (B, T, C).
        """
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.embed_dim, dim=2)                   # Linear transformation for queries, keys, and values

        # Reshape and transpose for scaled dot product attention of pytorch
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) 
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) 

        # Scaled Dot-Product Attention
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)                        # Reshape and apply linear projection
        y = self.resid_dropout(self.c_proj(y))

        return y

    
class MLP(torch.nn.Module):
    """
    Multi-Layer Perceptron (MLP) module.

    Args:
        embed_dim (int, optional): Dimension of input embeddings. Default is 768.
        bias (bool, optional): If True, learnable bias is added to the linear transformations. Default is True.
        dropout (float, optional): Dropout probability. Default is 0.1.

    Attributes:
        c_fc (torch.nn.Linear): Linear layer for the first fully connected transformation.
        gelu (torch.nn.GELU): GELU activation function.
        c_proj (torch.nn.Linear): Linear layer for the second fully connected transformation.
        dropout (torch.nn.Dropout): Dropout layer.

    Note:
        The input tensor `x` is expected to have dimensions (B, T, C), where
        B is the batch size, T is the sequence length, and C is the input embedding dimension.
    """
    def __init__(self, embed_dim:int=768, bias:bool=True, dropout=0.1) -> None:
        super(MLP, self).__init__()
        self.c_fc = torch.nn.Linear(embed_dim, 4*embed_dim, bias=bias)
        self.gelu = torch.nn.GELU()
        self.c_proj = torch.nn.Linear(4*embed_dim, embed_dim, bias=bias)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self,x):
        """
        Forward pass of the Multi-Layer Perceptron (MLP) module.

        Args:
            x (torch.Tensor): Input tensor with dimensions (B, T, C), where B is the batch size, T is the sequence length, and C is the input embedding dimension.

        Returns:
            torch.Tensor: Output tensor with dimensions (B, T, C).
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class DecoderBlock(torch.nn.Module):
    """
    Decoder Block module used in transformer architectures.

    Args:
        embed_dim (int): Dimension of input embeddings.
        n_heads (int): Number of attention heads.

    Attributes:
        ln_1 (torch.nn.LayerNorm): Layer normalization for the first sub-layer.
        attn (MultiheadAttention): Multihead Attention module.
        ln_2 (torch.nn.LayerNorm): Layer normalization for the second sub-layer.
        mlp (MLP): Multi-Layer Perceptron module.

    Note:
        The input tensor `x` is expected to have dimensions (B, T, C), where
        B is the batch size, T is the sequence length, and C is the input embedding dimension.
    """
    def __init__(self, embed_dim, n_heads) -> None:
        super(DecoderBlock, self).__init__()
        self.ln_1 = torch.nn.LayerNorm(embed_dim)                       # Layer normalization for the first sub-layer
        self.attn = MultiheadAttention(embed_dim, n_heads, True, 0.1)   # Multihead attention module
        self.ln_2 = torch.nn.LayerNorm(embed_dim)                       # Layer normalization for the second sub-layer
        self.mlp = MLP(embed_dim)                                       # Multilayer perceptron module
    
    def forward(self,x):
        """
        Forward pass of the Decoder Block module.

        Args:
            x (torch.Tensor): Input tensor with dimensions (B, T, C), where B is the batch size, T is the sequence length, and C is the input embedding dimension.

        Returns:
            torch.Tensor: Output tensor with dimensions (B, T, C).
        """
        x = x + self.attn(self.ln_1(x))                                 # First sub-layer: Multihead Attention with Residual connections
        x = x + self.mlp(self.ln_2(x))                                  # Second sub-layer: Multi-Layer Perceptron
        return x

class GPT2(torch.nn.Module):
    """
    GPT-2 model architecture.

    Args:
        embed_dim (int, optional): Dimension of input embeddings. Default is 768.
        n_layers (int, optional): Number of layers in the transformer. Default is 12.
        block_size (int, optional): Size of the blocks in the model. Default is 1024.
        vocab_size (int, optional): Size of the vocabulary. Default is 50257.
        n_heads (int, optional): Number of attention heads in the transformer blocks. Default is 12.

    Attributes:
        block_size (int): Size of the blocks in the model.
        transformer (torch.nn.ModuleDict): Module dictionary containing components of the GPT-2 model.
            - wte: Embedding layer for word tokens.
            - wpe: Embedding layer for positional encodings.
            - drop: Dropout layer.
            - h: List of DecoderBlocks forming the transformer layers.
            - ln_f: Layer normalization for the final layer.
        lm_head (torch.nn.Linear): Linear layer for language modeling prediction.

    Note:
        The input tensor `x` is expected to have dimensions (B, T), where
        B is the batch size and T is the sequence length.
    """
    def __init__(self, embed_dim:int=768, n_layers:int=12, block_size:int=1024, vocab_size:int=50257, n_heads:int=12) -> None:
        super().__init__()
        self.block_size = block_size                                    # Size of the blocks in the model
        self.transformer = torch.nn.ModuleDict(dict(                    # Model Architecture
            wte = torch.nn.Embedding(vocab_size, embed_dim),            # Word embeddings of size 50257 * 768 for vocobulary of size 50257 with each word represented by vector of 768 dimension
            wpe = torch.nn.Embedding(block_size, embed_dim),            # Position embeddings of size 1024 * 768 for max word token input of size 1024 with each token having 768 dimension representation
            drop = torch.nn.Dropout(0.1),
            h = torch.nn.ModuleList([DecoderBlock(embed_dim, n_heads) for _ in range(n_layers)]),# 12 decoder blocks for smallest GPT2
            ln_f = torch.nn.LayerNorm(embed_dim)
        ))
        self.lm_head = torch.nn.Linear(embed_dim, vocab_size, bias=False)# Linear layer for language modeling prediction

        self.transformer.wte.weight = self.lm_head.weight               # Tied weights
        self.apply(self._init_weights)                                  # Weight initialization

    def _init_weights(self, module):
        """
        Initialize weights for Linear and Embedding layers.

        Args:
            module (torch.nn.Module): The module to initialize.
        """
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Forward pass of the GPT-2 model.

        Args:
            x (torch.Tensor): Input tensor with dimensions (B, T), where B is the batch size and T is the sequence length.
            targets: Target token for training the model

        Returns:
            logits(torch.Tensor): Output tensor with dimensions (B, T, vocab_size).
            loss[Optional]: Calculated Loss
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        token_emb = self.transformer.wte(idx)                           # Create word embeddings
        pos_emb = self.transformer.wpe(pos)                             # Create positional embeddings
        x = self.transformer.drop(token_emb+pos_emb)                    # Create input by adding word and postional embeddings
        for block in self.transformer.h:
            x = block(x)                                                # Passing embeddings through decoder blocks to perform attention
        x = self.transformer.ln_f(x)
        # Getting logits for next word prediction
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens using the GPT-2 model.

        Args:
            idx (torch.Tensor): Initial sequence of tokens with dimensions (B, T), where B is the batch size and T is the sequence length.
            max_new_tokens (int): Maximum number of new tokens to generate.
            temperature (float, optional): Sampling temperature. Default is 1.0.
            top_k (int, optional): Number of top-k candidates to consider during sampling. Default is None.

        Returns:
            torch.Tensor: Tensor containing the generated sequence of tokens with dimensions (B, T + max_new_tokens).
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:] # Cropping the block if its bigger than 1024 tokens
            logits, _ = self(idx_cond)                                  # Getting logits for next word
            logits = logits[:, -1, :] / temperature

            if top_k is not None:                                       # Optionally filter logits with top-k sampling
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)                           # Getting probability from logits
            idx_next = torch.multinomial(probs, num_samples=1)          # Getting next token
            idx = torch.cat((idx, idx_next), dim=1)                     # Appending next token to already existing sequence

        return idx
    
    @classmethod
    def from_pretrained(cls, override_args=None):
        """
        Instantiate a GPT2 model from pre-trained weights.

        Args:
            override_args (dict, optional): Dictionary of arguments to override default configuration.
                Currently, only 'dropout' is supported. Default is None.

        Returns:
            GPT2: An instance of the GPT2 model with pre-trained weights.
        """
        override_args = override_args or {}
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel

        # Create an instance of the coded GPT2 model
        coded_model = GPT2()
        sd = coded_model.state_dict()
        coded_model_keys = sd.keys()
        coded_model_keys = [k for k in coded_model_keys if not k.endswith('.attn.bias')]

        # Load the pre-trained GPT2LMHeadModel
        loaded_model = GPT2LMHeadModel.from_pretrained('gpt2')
        loaded_sd = loaded_model.state_dict()

        loaded_model_keys = loaded_sd.keys()
        loaded_model_keys = [k for k in loaded_model_keys if not k.endswith('.attn.masked_bias')] 
        loaded_model_keys = [k for k in loaded_model_keys if not k.endswith('.attn.bias')] 
        
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        # Transfer weights from the pre-trained model to the coded model
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