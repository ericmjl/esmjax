from typing import Optional

import haiku as hk
import jax.numpy as jnp

from .layers import TransformerLayer, LearnedPositionalEmbeddings, ContactPredHead

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
