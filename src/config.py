import jax.numpy as jnp

MODEL_CONFIG = {
    'num_layers': 6,
    'embed_dim': 512,
    'num_heads': 8,
    'mlp_features': 2048,
    'dropout_rate': 0.1,
    'vocab_size': 10000,
    'max_len': 128,
    'dtype': jnp.float32
}

TRAINING_CONFIG = {
    'num_epochs': 10,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'warmup_steps': 1000,
    'batches_per_epoch': 10
}