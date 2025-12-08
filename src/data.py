import jax
import jax.numpy as jnp

def generate_dummy_batch(batch_key, batch_size, enc_len, dec_len, vocab_size):
    """
    Generates synthetic data for demonstration purposes.
    """
    enc_input = jax.random.randint(batch_key, (batch_size, enc_len), minval=1, maxval=vocab_size)
    dec_input = jax.random.randint(batch_key, (batch_size, dec_len), minval=1, maxval=vocab_size)
    
    labels = jnp.concatenate([dec_input[:, 1:], jnp.zeros((batch_size, 1), dtype=jnp.int32)], axis=1)
    
    enc_input = enc_input.at[:, enc_len // 2:].set(0)
    dec_input = dec_input.at[:, dec_len // 2:].set(0)
    labels = labels.at[:, dec_len // 2:].set(0)

    return {
        'encoder_input_ids': enc_input,
        'decoder_input_ids': dec_input,
        'labels': labels,
        'vocab_size': vocab_size
    }