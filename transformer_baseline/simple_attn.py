# %%
import base_layers as au
import torch
from torch import nn


def warn_nan(x, msg=''):
    if torch.isnan(x).any():
        print(msg, "is nan")


class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.dim = dim
        self.inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(0, max_seq_len, dtype=torch.float)
        sinusoid_inp = torch.einsum("i,j->ij", position, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        self.register_buffer('emb', emb)
    
    def calculate_new_embed(self, x):
        position = torch.arange(0, x.shape[1], dtype=torch.float)
        sinusoid_inp = torch.einsum("i,j->ij", position, self.inv_freq)
        return torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1).unsqueeze(0).to(x)
    
    def forward(self, x):
        return self.emb[None, :x.shape[1], :].to(x)


class DecoderBlock(nn.Module):
    def __init__(self, heads, n_state, proj_forward, activation, dropout):
        super().__init__()
        self.heads = heads
        self.n_state = n_state

        self.attn = torch.nn.MultiheadAttention(n_state, heads, dropout, bias=True, batch_first=True)

        self.linear1 = nn.Linear(n_state, proj_forward)
        self.dropout = nn.Dropout(dropout)

        self.linear2 = nn.Linear(proj_forward, n_state)
        self.norm1 = nn.LayerNorm(n_state)
        self.norm2 = nn.LayerNorm(n_state)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation
    
    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def forward(self, x, mask, pos=None):
        q = k = self.with_pos_embed(x, pos)
        x1 = self.attn(q, k, value=x, attn_mask=mask)[0]
        warn_nan(x1, "attn")
        x = x + self.dropout1(x1)
        x = self.norm1(x)
        x1 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(x1)
        x = self.norm2(x)
        warn_nan(x, 'feed forward')
        return x


class Transformer(nn.Module):
    def __init__(self, 
        vocab=618,
        embed_dim=512,
        n_layers=6, n_heads=8, max_sequence=1024, proj_forward=4096, dropout=0.1, 
        activation='gelu',
        **kwargs) -> None:
        super().__init__()

        self.vocab = vocab
        self.d_model = embed_dim
        self.n_layers = n_layers
        self.heads = n_heads
        self.max_sequence = max_sequence
        
        self.embed = nn.Embedding(vocab, embed_dim)

        activation = getattr(torch.nn.functional, activation)
        self.decoder_layers = nn.ModuleList([DecoderBlock(n_heads, self.d_model, proj_forward, activation, dropout) for _ in range(self.n_layers)])
        self.norm = nn.LayerNorm(self.d_model)

        self.pos_emb = FixedPositionalEmbedding(self.d_model, max_sequence)
        mask = au.generate_square_subsequent_mask(max_sequence, 'cpu')
        self.register_buffer("mask", mask)
    
    def forward(self, x: torch.Tensor):
        _, seq_len = x.size()
        h = self.embed(x)

        if seq_len > self.max_sequence:
            mask = au.generate_square_subsequent_mask(seq_len, x.device)
            pos = self.pos_emb.calculate_new_embed(x)
        else:
            mask = self.mask[:seq_len, :seq_len]
            pos = self.pos_emb(x)

        for layer in range(self.n_layers):
            h = self.decoder_layers[layer](h, mask, pos)
            warn_nan(h, f'{layer} layer')

        h = self.norm(h)
        warn_nan(h, 'norm')
        logits = h @ self.embed.weight.transpose(-1, -2)
        warn_nan(logits, 'logit')

        return logits
    
