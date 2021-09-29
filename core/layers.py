from typing import Optional

import haiku as hk
import jax.numpy as jnp
import jax

class SelfMultiHeadAttention(hk.MultiHeadAttention):
    """Self attention with mask applied"""
    
    def __call__(
        self, 
        x: jnp.ndarray,
        att_mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:

        return super().__call__(query=x, key=x, value=x, mask=att_mask)


class TransformerLayer(hk.Module):
    def __init__(self, 
                num_heads:int, 
                key_size:int, 
                name: Optional[str]=None):

        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
    
    def __call__(
        self, 
        x: jnp.ndarray,
        att_mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:

        res = x 

        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="self_attn_layer_norm")(x)
        x = SelfMultiHeadAttention(self.num_heads, self.key_size, 1)(x, att_mask)
        x = res + x 

        res = x 
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="final_layer_norm")(x)
        x = jax.nn.gelu(hk.Linear(self.num_heads * self.key_size * 4, name="fc1")(x), approximate=False)
        x = hk.Linear(self.num_heads * self.key_size, name="fc2")(x)
        x = res + x

        return x

class LearnedPositionalEmbeddings(hk.Module):
    def __init__(self, 
                vocab_size: int,
                embed_dim: int,
                padding_idx: Optional[int] = 1,
                name: Optional[str] = None):

        super().__init__(name=name)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx
    
    def __call__(self, tokens: jnp.ndarray) -> jnp.ndarray:
        mask = tokens != self.padding_idx
        positions = jnp.cumsum(mask, axis=1) * mask + self.padding_idx

        return hk.Embed(self.vocab_size + self.padding_idx + 1, self.embed_dim)(positions)

class ESM1b(hk.Module):
    def __init__(self,
                padding_idx: Optional[int] = 1,
                name: Optional[str] = None):
        
        self.padding_idx = padding_idx
        super().__init__(name=name)
    
    def __call__(self, tokens: jnp.ndarray) -> jnp.ndarray:
        padding_mask = tokens == self.padding_idx

        att_mask = ~padding_mask[:, None, :]
        att_mask = jnp.einsum("bhT, bht->bhtT", att_mask, att_mask)

        x = hk.Embed(33, 1280)(tokens)
        x = x + LearnedPositionalEmbeddings(1024, 1280)(tokens)

        return x, att_mask