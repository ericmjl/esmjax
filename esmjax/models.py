from typing import Optional

import haiku as hk
import jax.numpy as jnp

from .layers import TransformerLayer, LearnedPositionalEmbeddings, ContactPredHead

class ESM1b(hk.Module):
    """Creates an ESM1b protein language model, ready for 
    inference."""
    def __init__(self,
        padding_idx: Optional[int] = 1,
        eos_idx: Optional[int] = 2,
        head_weights: Optional[bool] = False,
        contacts: Optional[bool] = False,
        name: Optional[str] = None):
        """Initializes the ESM1b model

        Args:
            padding_idx (Optional[int]): Token that corresponds to <pad>. Defaults to 1.
            eos_idx (Optional[int]): Token that corresponds to <eos>. Defaults to 2.
            head_weights (Optional[bool]): Whether to return head weights. Defaults to False.
            contacts (Optional[bool]): Whether to return contact predictions. Defaults to False.
            name (Optional[str]): Name for the module (custom will break weight loading). Defaults to None.
        """
        
        self.padding_idx = padding_idx
        self.eos_idx = eos_idx
        self.head_weights = head_weights or contacts
        self.contacts = contacts
        super().__init__(name=name)
    
    def __call__(self, 
        tokens: jnp.ndarray
        ) -> "dict[str, jnp.ndarray]":
        padding_mask_tokens = tokens == self.padding_idx

        padding_mask_att = ~padding_mask_tokens[:, None, :]
        padding_mask_att = jnp.einsum("bhT, bht->bhtT", padding_mask_att, padding_mask_att)

        #---------

        x = hk.Embed(33, 1280)(tokens)

        # 0.88 to undo RoBERTa's mask scaling factor
        x = 0.88 * x + LearnedPositionalEmbeddings(1024, 1280)(tokens)

        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="emb_layer_norm_before")(x)

        all_attn_weights = [] if self.head_weights else None
        contacts = None

        # construct a tower of 33 transformer layers
        for i in range(33):
            # 20 heads, each with 64 dim Q,K,V triplets
            output = TransformerLayer(20, 64, name=f"transformer_layer_{i}")(x, padding_mask_att)
            x = output["x"]

            if self.head_weights:
                all_attn_weights.append(output["attn_weights"])
        #----------
        # mask all <cls>, <eos> and <pad> embeddings, head_weights and contact_predictions
        mask_tokens = (tokens == self.eos_idx) | (tokens == 0) | (tokens == self.padding_idx) # B x T

        # if head_weights was requested, stack them and mask out
        if self.head_weights or self.contacts:
            mask = ~mask_tokens[:, None, None, :] # B x 1 x 1 x T
            mask = jnp.einsum("blhT, blht->blhtT", mask, mask) # B x 1 x 1 x T x T

            all_attn_weights = jnp.stack(all_attn_weights, axis=1)
            all_attn_weights = all_attn_weights * mask

            if self.contacts:
                contacts = ContactPredHead()(all_attn_weights) * mask[:, 0, 0, :, :]

        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="emb_layer_norm_after")(x)
        
        return {"embeddings": x * ~mask_tokens[:, :, None], 
                "head_weights": all_attn_weights,
                "contacts": contacts}