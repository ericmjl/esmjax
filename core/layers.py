from typing import Optional, Tuple

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

    def __call__(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
    ) -> "dict[str, jnp.ndarray]":

        attn_weights = self.attn_weights(x, mask)
        x = self.compute_tokens(x, attn_weights)
        return {"x": x, "attn_weights": attn_weights}
        
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
        name: Optional[str]=None):

        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
    
    def __call__(
        self, 
        x: jnp.ndarray,
        att_mask: Optional[jnp.ndarray] = None
        ) -> "dict[str, jnp.ndarray]":

        res = x 

        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="self_attn_layer_norm")(x)
        output = SelfMHA(self.num_heads, self.key_size, 1)(x, att_mask)
        x = output["x"]
        x = res + x 

        res = x 
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="final_layer_norm")(x)
        x = jax.nn.gelu(hk.Linear(self.num_heads * self.key_size * 4, name="fc1")(x), approximate=False)
        x = hk.Linear(self.num_heads * self.key_size, name="fc2")(x)
        x = res + x
        
        output["x"] = x
        return output

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
    
    def __call__(self, 
        tokens: jnp.ndarray
        ) -> jnp.ndarray:
        mask = tokens != self.padding_idx
        positions = jnp.cumsum(mask, axis=1) * mask + self.padding_idx

        return hk.Embed(self.vocab_size + self.padding_idx + 1, self.embed_dim)(positions)

class ContactPredHead(hk.Module):
    def __init__(self,
        padding_idx: Optional[int] = 1,
        eos_idx: Optional[int] = 2,
        name: Optional[str] = None):

        self.padding_idx = padding_idx
        self.eos_idx = eos_idx
        super().__init__(name=name)

    def __call__(self, 
        tokens: jnp.ndarray,
        head_weights: jnp.ndarray
        ) -> jnp.ndarray:

        mask_tokens = (tokens == self.eos_idx) | (tokens == 0) | (tokens == self.padding_idx) # B x T
        mask = ~mask_tokens[:, None, None, :] # B x L x H x T
        mask = jnp.einsum("blhT, blht->blhtT", mask, mask) # B x L x H x T x T
        head_weights = head_weights * mask
        
        head_weights = self.avg_prod_correct(self.symmetrize(head_weights))
        B, layers, heads, T, _ = head_weights.shape
        head_weights = head_weights.reshape(B, layers*heads, T, T) 
        head_weights = jnp.transpose(head_weights, axes=(0, 2, 3, 1))
        
        contact_preds = jax.nn.sigmoid(hk.Linear(1, name="regression")(head_weights))
        contact_preds = contact_preds[:, :, :, 0] # last index has dim=1

        return contact_preds * mask[:, 0, 0, :, :]

    @staticmethod
    def avg_prod_correct(x):
        a1 = x.sum(axis=-1, keepdims=True)
        a2 = x.sum(axis=-2, keepdims=True)

        a12 = x.sum(axis=(-1, -2), keepdims=True)

        avg = (a1 * a2) / a12 
        normalized = x - avg 

        return normalized
    
    @staticmethod
    def symmetrize(x):
        # symmetrize along token x token dimensions
        return x + jnp.transpose(x, axes=(0, 1, 2, 4, 3))

class ESM1b(hk.Module):
    def __init__(self,
        padding_idx: Optional[int] = 1,
        head_weights: Optional[bool] = False,
        name: Optional[str] = None):
        
        self.padding_idx = padding_idx
        self.head_weights = head_weights
        super().__init__(name=name)
    
    def __call__(self, 
        tokens: jnp.ndarray
        ) -> "dict[str, jnp.ndarray]":
        padding_mask = tokens == self.padding_idx

        att_mask = ~padding_mask[:, None, :]
        att_mask = jnp.einsum("bhT, bht->bhtT", att_mask, att_mask)

        x = hk.Embed(33, 1280)(tokens)

        # 0.88 to undo RoBERTa's mask scaling factor
        x = 0.88 * x + LearnedPositionalEmbeddings(1024, 1280)(tokens)

        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="emb_layer_norm_before")(x)

        all_attn_weights = []

        for i in range(33):
            output = TransformerLayer(20, 64, name=f"transformer_layer_{i}")(x, att_mask)
            x = output["x"]

            if self.head_weights:
                all_attn_weights.append(output["attn_weights"])

        if self.head_weights:
            all_attn_weights = jnp.stack(all_attn_weights, axis=1)
            all_attn_weights = all_attn_weights * att_mask[:, None, :, :, :]

        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="emb_layer_norm_after")(x)
        
        return {"embeddings": x, "head_weights": all_attn_weights}

class ESM1bContactPredictor(hk.Module):
    def __init__(self,
        padding_idx: Optional[int] = 1,
        eos_idx: Optional[int] = 2,
        name: Optional[str] = None
        ):

        self.padding_idx = padding_idx
        self.eos_idx = eos_idx
        super().__init__(name=name)
    
    @hk.transparent
    def __call__(self, 
        tokens: jnp.ndarray
        ) -> jnp.ndarray:
        
        output = ESM1b(head_weights=True, name="esm1b")(tokens)
        return ContactPredHead()(tokens, output["head_weights"])
