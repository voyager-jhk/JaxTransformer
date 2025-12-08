import jax.numpy as jnp
import flax.linen as nn

def create_masks(encoder_input_tokens: jnp.ndarray, decoder_input_tokens: jnp.ndarray):
    """
    生成 Encoder Padding Mask, Decoder Self-Attention Mask (Causal), 和 Cross-Attention Mask
    """
    # 1. Encoder Padding Mask (batch, 1, 1, enc_len)
    encoder_padding_mask = (encoder_input_tokens != 0)[..., None, None, :]
    
    # 2. Decoder Self-Attention Mask (batch, 1, dec_len, dec_len)
    # 结合 Causal Mask (下三角) 和 Padding Mask
    decoder_padding_mask = (decoder_input_tokens != 0)[..., None, None, :]
    causal_mask = nn.make_causal_mask(decoder_input_tokens)
    decoder_self_attention_mask = nn.combine_masks(causal_mask, decoder_padding_mask)

    # 3. Cross-Attention Mask (batch, 1, 1, enc_len)
    # Decoder 关注 Encoder 输出时，需要忽略 Encoder 的 Padding
    cross_attention_mask = (encoder_input_tokens != 0)[..., None, None, :]

    return encoder_padding_mask, decoder_self_attention_mask, cross_attention_mask