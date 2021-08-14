from typing import Optional

import haiku as hk
import jax.numpy as jnp
import jax

class SelfMultiHeadAttention(hk.MultiHeadAttention):
    """Self attention with no mask applied"""
    
    def __call__(
        self, 
        x: jnp.ndarray) -> jnp.ndarray:

        return super().__call__(query=x, key=x, value=x)


class TransformerLayer(hk.Module):
    def __init__(self, num_heads:int, key_size:int, name: Optional[str]=None):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
    
    def __call__(
        self, 
        x: jnp.ndarray) -> jnp.ndarray:

        res = x 

        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="self_attn_layer_norm")(x)
        x = SelfMultiHeadAttention(self.num_heads, self.key_size, 1)(x)
        x = res + x 

        res = x 
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="final_layer_norm")(x)
        x = jax.nn.gelu(hk.Linear(self.num_heads * self.key_size * 4, name="fc1")(x), approximate=False)
        x = hk.Linear(self.num_heads * self.key_size, name="fc2")(x)
        x = res + x

        return x
