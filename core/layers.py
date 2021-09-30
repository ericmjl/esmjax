from typing import Optional

from jax.interpreters.xla import xla_primitive_callable

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


# Class is modified from MultiHeadAttention here
# https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/attention.py
# Full original Apache 2.0 license here https://github.com/deepmind/dm-haiku/blob/main/LICENSE
class SelfMHA(hk.MultiHeadAttention):
    """Self multi-head attention with mask applied. Modified from 
    original implementation to be able to return attention
    weights (needed for contact prediction)"""
    
    def __init__(
        self,
        num_heads: int,
        key_size: int,
        w_init_scale: float,
        head_weights: Optional[bool] = False,
        value_size: Optional[int] = None,
        model_size: Optional[int] = None,
        name: Optional[str] = None,
    ):
        super().__init__(num_heads,
                        key_size,
                        w_init_scale,
                        value_size,
                        model_size,
                        name)

        self.head_weights = head_weights

    def __call__(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:

        attn_weights = self.attn_weights(x, mask)
        return self.compute_tokens(x, attn_weights)
        
    @hk.transparent
    def attn_weights(
        self, 
        x: jnp.ndarray, 
        mask: Optional[jnp.ndarray] = None,
        ) -> jnp.ndarray:
        query_heads = self._linear_projection(x, self.key_size, "query")
        key_heads = self._linear_projection(x, self.key_size, "key")

        attn_logits = jnp.einsum("...thd,...Thd->...htT", query_heads, key_heads)
        sqrt_key_size = np.sqrt(self.key_size).astype(x.dtype)
        attn_logits = attn_logits / sqrt_key_size

        if mask is not None:
            assert len(mask.shape) == len(attn_logits.shape)
            attn_logits = jnp.where(mask, attn_logits, -1e30)

        attn_weights = jax.nn.softmax(attn_logits)

        return attn_weights
    
    @hk.transparent
    def compute_tokens(
        self,
        x: jnp.ndarray,
        attn_weights: jnp.ndarray,
    ) -> jnp.ndarray:
        value_heads = self._linear_projection(x, self.value_size, "value")
        attn = jnp.einsum("...htT,...Thd->...thd", attn_weights, value_heads)
        
        # Concatenate attention matrix of all heads into a single vector.
        attn_vec = jnp.reshape(attn, (*x.shape[:-1], -1))
        return hk.Linear(self.model_size, w_init=self.w_init)(attn_vec)

class TransformerLayer(hk.Module):
    """Each of the blocks that form the 33-layer tower"""
    def __init__(self, 
                num_heads:int, 
                key_size:int, 
                head_weights: Optional[bool] = False,
                name: Optional[str]=None):

        super().__init__(name=name)
        self.num_heads = num_heads
        self.head_weights = head_weights
        self.key_size = key_size
    
    def __call__(
        self, 
        x: jnp.ndarray,
        att_mask: Optional[jnp.ndarray] = None
        ) -> jnp.ndarray:

        res = x 

        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="self_attn_layer_norm")(x)
        x = SelfMHA(self.num_heads, self.key_size, 1, self.head_weights)(x, att_mask)

        x = res + x 

        res = x 
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="final_layer_norm")(x)
        x = jax.nn.gelu(hk.Linear(self.num_heads * self.key_size * 4, name="fc1")(x), approximate=False)
        x = hk.Linear(self.num_heads * self.key_size, name="fc2")(x)
        x = res + x

        return x

class LearnedPositionalEmbeddings(hk.Module):
    """Position embeddings to be added to token embeddings"""
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
    
    def __call__(self, 
                tokens: jnp.ndarray
                ) -> jnp.ndarray:
        padding_mask = tokens == self.padding_idx

        att_mask = ~padding_mask[:, None, :]
        att_mask = jnp.einsum("bhT, bht->bhtT", att_mask, att_mask)

        x = hk.Embed(33, 1280)(tokens)

        # 0.88 to undo RoBERTa's mask scaling factor
        x = 0.88 * x + LearnedPositionalEmbeddings(1024, 1280)(tokens)

        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="emb_layer_norm_before")(x)

        for i in range(33):
            x = TransformerLayer(20, 64, name=f"transformer_layer_{i}")(x, att_mask)

        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="emb_layer_norm_after")(x)
        
        return x
