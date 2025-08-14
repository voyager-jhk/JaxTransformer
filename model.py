import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
from flax.training import common_utils
from typing import Sequence, Callable, Any, Optional

# --- 1. 核心组件定义 ---

# 1.1 Positional Encoding
class PositionalEncoding(nn.Module):
    max_len: int
    embed_dim: int

    @nn.compact
    def __call__(self, inputs: jnp.ndarray):
        # inputs: (batch_size, seq_len, embed_dim)
        seq_len = inputs.shape[1]
        
        # Create positional encodings
        position = jnp.arange(self.max_len, dtype=jnp.float32)[jnp.newaxis, :] # (1, max_len)
        div_term = jnp.exp(jnp.arange(0, self.embed_dim, 2) * -(jnp.log(10000.0) / self.embed_dim)) # (embed_dim/2,)

        pe = jnp.zeros((self.max_len, self.embed_dim))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        pe = pe[jnp.newaxis, :seq_len, :] # (1, seq_len, embed_dim)

        return inputs + jnp.array(pe, dtype=inputs.dtype) # Add to embeddings

# 1.2 Multi-Layer Perceptron (FFN)
class MLP(nn.Module):
    features: int # Inner dimension
    out_features: int # Output dimension
    dropout_rate: float = 0.1
    deterministic: bool = False # For dropout

    @nn.compact
    def __call__(self, inputs: jnp.ndarray):
        x = nn.Dense(self.features, name='wi')(inputs)
        x = nn.relu(x)
        x = nn.Dropout(self.dropout_rate, deterministic=self.deterministic)(x)
        x = nn.Dense(self.out_features, name='wo')(x)
        return x

# 1.3 Multi-Head Attention Wrapper
# This wraps nn.MultiHeadDotProductAttention to simplify usage within blocks
class SelfAttention(nn.Module):
    num_heads: int
    qkv_features: int
    out_features: int
    dropout_rate: float
    deterministic: bool
    dtype: Any = jnp.float32 # Data type for attention weights (e.g., jnp.float32 or jnp.bfloat16)

    @nn.compact
    def __call__(self, inputs_q, inputs_kv, mask=None):
        query = nn.Dense(self.qkv_features, name='query_proj')(inputs_q)
        key = nn.Dense(self.qkv_features, name='key_proj')(inputs_kv)
        value = nn.Dense(self.qkv_features, name='value_proj')(inputs_kv)

        attention_output = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dtype=self.dtype,
            deterministic=self.deterministic,
            dropout_rate=self.dropout_rate,
            name='attention_core'
        )(query, key, value, mask)
        
        output = nn.Dense(self.out_features, name='output_proj')(attention_output)
        return output

# 1.4 Encoder Block
class EncoderBlock(nn.Module):
    embed_dim: int
    num_heads: int
    qkv_features: int
    mlp_features: int
    dropout_rate: float = 0.1
    deterministic: bool = False
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, mask: jnp.ndarray = None):
        # Self-Attention sub-layer
        norm_inputs = nn.LayerNorm(name='norm1')(inputs)
        attention_output = SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.qkv_features,
            out_features=self.embed_dim,
            dropout_rate=self.dropout_rate,
            deterministic=self.deterministic,
            dtype=self.dtype,
            name='self_attention'
        )(norm_inputs, norm_inputs, mask)
        
        # Add & Norm (Residual connection and Layer Normalization)
        x = inputs + attention_output

        # MLP sub-layer
        norm_x = nn.LayerNorm(name='norm2')(x)
        mlp_output = MLP(
            features=self.mlp_features,
            out_features=self.embed_dim,
            dropout_rate=self.dropout_rate,
            deterministic=self.deterministic,
            name='mlp'
        )(norm_x)

        # Add & Norm
        output = x + mlp_output
        return output

# 1.5 Decoder Block
class DecoderBlock(nn.Module):
    embed_dim: int
    num_heads: int
    qkv_features: int
    mlp_features: int
    dropout_rate: float = 0.1
    deterministic: bool = False
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, encoder_outputs: jnp.ndarray,
                 self_attention_mask: jnp.ndarray = None,
                 cross_attention_mask: jnp.ndarray = None):
        
        # Masked Self-Attention sub-layer
        norm_inputs = nn.LayerNorm(name='norm1')(inputs)
        self_attention_output = SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.qkv_features,
            out_features=self.embed_dim,
            dropout_rate=self.dropout_rate,
            deterministic=self.deterministic,
            dtype=self.dtype,
            name='self_attention'
        )(norm_inputs, norm_inputs, self_attention_mask)
        
        # Add & Norm
        x = inputs + self_attention_output

        # Cross-Attention sub-layer
        norm_x_cross = nn.LayerNorm(name='norm2')(x)
        cross_attention_output = SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.qkv_features,
            out_features=self.embed_dim,
            dropout_rate=self.dropout_rate,
            deterministic=self.deterministic,
            dtype=self.dtype,
            name='cross_attention'
        )(norm_x_cross, encoder_outputs, cross_attention_mask) # Query from decoder, Key/Value from encoder
        
        # Add & Norm
        x = x + cross_attention_output

        # MLP sub-layer
        norm_x_mlp = nn.LayerNorm(name='norm3')(x)
        mlp_output = MLP(
            features=self.mlp_features,
            out_features=self.embed_dim,
            dropout_rate=self.dropout_rate,
            deterministic=self.deterministic,
            name='mlp'
        )(norm_x_mlp)

        # Add & Norm
        output = x + mlp_output
        return output

# 1.6 Transformer Encoder Stack
class TransformerEncoder(nn.Module):
    num_layers: int
    embed_dim: int
    num_heads: int
    qkv_features: int
    mlp_features: int
    dropout_rate: float = 0.1
    deterministic: bool = False
    vocab_size: int = None
    max_len: int = 512
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, input_tokens: jnp.ndarray, attention_mask: jnp.ndarray = None):
        # input_tokens: (batch_size, seq_len) token IDs
        
        # Token Embeddings
        x = nn.Embed(num_embeddings=self.vocab_size, features=self.embed_dim, name='token_embeddings')(input_tokens)
        
        # Positional Encoding
        x = PositionalEncoding(max_len=self.max_len, embed_dim=self.embed_dim, name='pos_enc')(x)
        x = nn.Dropout(self.dropout_rate, deterministic=self.deterministic)(x) # Dropout after summing embeddings and positional enc.

        # Stack Encoder Blocks
        for i in range(self.num_layers):
            x = EncoderBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                qkv_features=self.qkv_features,
                mlp_features=self.mlp_features,
                dropout_rate=self.dropout_rate,
                deterministic=self.deterministic,
                dtype=self.dtype,
                name=f'encoder_block_{i}'
            )(x, attention_mask)
        
        # Final Layer Norm (optional, but common)
        output = nn.LayerNorm(name='encoder_final_norm')(x)
        return output

# 1.7 Transformer Decoder Stack
class TransformerDecoder(nn.Module):
    num_layers: int
    embed_dim: int
    num_heads: int
    qkv_features: int
    mlp_features: int
    dropout_rate: float = 0.1
    deterministic: bool = False
    vocab_size: int = None
    max_len: int = 512
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, target_tokens: jnp.ndarray, encoder_outputs: jnp.ndarray,
                 decoder_self_attention_mask: jnp.ndarray = None,
                 cross_attention_mask: jnp.ndarray = None):
        
        # Token Embeddings
        x = nn.Embed(num_embeddings=self.vocab_size, features=self.embed_dim, name='token_embeddings')(target_tokens)
        
        # Positional Encoding
        x = PositionalEncoding(max_len=self.max_len, embed_dim=self.embed_dim, name='pos_enc')(x)
        x = nn.Dropout(self.dropout_rate, deterministic=self.deterministic)(x)

        # Stack Decoder Blocks
        for i in range(self.num_layers):
            x = DecoderBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                qkv_features=self.qkv_features,
                mlp_features=self.mlp_features,
                dropout_rate=self.dropout_rate,
                deterministic=self.deterministic,
                dtype=self.dtype,
                name=f'decoder_block_{i}'
            )(x, encoder_outputs, decoder_self_attention_mask, cross_attention_mask)
        
        # Final Layer Norm (optional)
        x = nn.LayerNorm(name='decoder_final_norm')(x)

        # Output projection to vocabulary probabilities (logits)
        output = nn.Dense(self.vocab_size, name='logits')(x)
        return output

# 1.8 Full Transformer Model (Encoder-Decoder)
class Transformer(nn.Module):
    encoder_num_layers: int
    decoder_num_layers: int
    embed_dim: int
    num_heads: int
    qkv_features: int
    mlp_features: int
    dropout_rate: float = 0.1
    deterministic: bool = False
    vocab_size: int = None
    max_len: int = 512
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, encoder_input_tokens: jnp.ndarray, decoder_input_tokens: jnp.ndarray,
                 encoder_attention_mask: jnp.ndarray = None,
                 decoder_self_attention_mask: jnp.ndarray = None,
                 cross_attention_mask: jnp.ndarray = None):
        
        # Encoder
        encoder_outputs = TransformerEncoder(
            num_layers=self.encoder_num_layers,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            qkv_features=self.qkv_features,
            mlp_features=self.mlp_features,
            dropout_rate=self.dropout_rate,
            deterministic=self.deterministic,
            vocab_size=self.vocab_size,
            max_len=self.max_len,
            dtype=self.dtype,
            name='encoder'
        )(encoder_input_tokens, encoder_attention_mask)
        
        # Decoder
        decoder_outputs = TransformerDecoder(
            num_layers=self.decoder_num_layers,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            qkv_features=self.qkv_features,
            mlp_features=self.mlp_features,
            dropout_rate=self.dropout_rate,
            deterministic=self.deterministic,
            vocab_size=self.vocab_size,
            max_len=self.max_len,
            dtype=self.dtype,
            name='decoder'
        )(decoder_input_tokens, encoder_outputs, decoder_self_attention_mask, cross_attention_mask)
        
        return decoder_outputs # Usually the logits from the decoder for sequence generation

# --- 2. 掩码生成 ---
# 假设 padding token ID 为 0
def create_masks(encoder_input_tokens: jnp.ndarray, decoder_input_tokens: jnp.ndarray):
    """
    Generates attention masks for encoder, decoder self-attention, and cross-attention.
    Args:
        encoder_input_tokens: (batch_size, enc_seq_len)
        decoder_input_tokens: (batch_size, dec_seq_len)
    Returns:
        encoder_attention_mask: (batch_size, 1, 1, enc_seq_len) for padding
        decoder_self_attention_mask: (batch_size, 1, dec_seq_len, dec_seq_len) for causal + padding
        cross_attention_mask: (batch_size, 1, 1, enc_seq_len) for encoder padding
    """
    # Encoder attention mask (for padding)
    # This creates a boolean mask where True means 'attend' (non-padded), False means 'mask' (padded)
    # The mask shape is (batch, 1, 1, seq_len) to be broadcastable to attention scores (batch, heads, seq_len, seq_len)
    encoder_padding_mask = (encoder_input_tokens != 0)[..., None, None, :]
    
    # Decoder self-attention mask (for causal and padding)
    decoder_padding_mask = (decoder_input_tokens != 0)[..., None, None, :]
    causal_mask = nn.make_causal_mask(decoder_input_tokens) # (batch, 1, dec_seq_len, dec_seq_len)
    # Combine causal mask (preventing future attention) with padding mask (preventing attention to padding)
    decoder_self_attention_mask = nn.combine_masks(causal_mask, decoder_padding_mask)

    # Cross-attention mask (decoder attends to encoder outputs, considering encoder padding)
    # The query sequence length is implicit in the attention operation, here we mask keys/values.
    # Shape needs to be broadcastable: (batch, 1, 1, enc_seq_len)
    cross_attention_mask = (encoder_input_tokens != 0)[..., None, None, :]

    return encoder_padding_mask, decoder_self_attention_mask, cross_attention_mask

# --- 3. 训练准备 ---

# 3.1 训练状态类
class TrainState(train_state.TrainState):
    # 将 dropout 的 PRNGKey 存储在 TrainState 中，以便在每次训练步中更新
    dropout_rng: jax.Array

# 3.2 损失函数 (交叉熵)
@jax.jit
def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray, label_smoothing: float = 0.0):
    """
    Calculates sparse cross-entropy loss with optional label smoothing.
    Handles padding by masking out loss for label ID 0.
    """
    num_classes = logits.shape[-1]
    one_hot_labels = common_utils.onehot(labels, num_classes=num_classes)
    
    # Apply label smoothing
    if label_smoothing > 0.0:
        smooth_labels = one_hot_labels * (1.0 - label_smoothing) + \
                        label_smoothing / num_classes
        loss = -jnp.sum(smooth_labels * jax.nn.log_softmax(logits), axis=-1)
    else:
        loss = -jnp.sum(one_hot_labels * jax.nn.log_softmax(logits), axis=-1)
    
    # Mask out loss for padding tokens (assuming 0 is padding ID for labels)
    padding_mask = (labels != 0).astype(jnp.float32)
    loss = loss * padding_mask
    
    # Sum over sequence and average over non-padded elements
    return jnp.sum(loss) / jnp.sum(padding_mask)

# 3.3 单个训练步骤
@jax.jit
def train_step(state: TrainState, batch: dict, learning_rate: float):
    """Performs a single training step."""
    # Split dropout RNG for the current step
    dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

    def loss_fn(params):
        # Generate masks for the batch
        enc_mask, dec_self_mask, cross_mask = create_masks(
            batch['encoder_input_ids'], batch['decoder_input_ids']
        )
        
        # Model forward pass
        logits = Transformer(
            encoder_num_layers=state.apply_fn.keywords['encoder_num_layers'],
            decoder_num_layers=state.apply_fn.keywords['decoder_num_layers'],
            embed_dim=state.apply_fn.keywords['embed_dim'],
            num_heads=state.apply_fn.keywords['num_heads'],
            qkv_features=state.apply_fn.keywords['qkv_features'],
            mlp_features=state.apply_fn.keywords['mlp_features'],
            dropout_rate=state.apply_fn.keywords['dropout_rate'],
            deterministic=False, # Set to False during training to enable dropout
            vocab_size=state.apply_fn.keywords['vocab_size'],
            max_len=state.apply_fn.keywords['max_len'],
            dtype=state.apply_fn.keywords['dtype'],
        ).apply(
            {'params': params}, # Apply with current parameters
            batch['encoder_input_ids'],
            batch['decoder_input_ids'],
            encoder_attention_mask=enc_mask,
            decoder_self_attention_mask=dec_self_mask,
            cross_attention_mask=cross_mask,
            rngs={'dropout': dropout_rng} # Pass dropout RNG to the model
        )
        # Calculate loss
        loss = cross_entropy_loss(logits, batch['labels'])
        return loss, logits # Return logits for metrics/debugging

    # Calculate loss and gradients
    # jax.value_and_grad returns (value, gradients)
    # has_aux=True means loss_fn also returns auxiliary data (logits in this case)
    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    
    # Apply gradient clipping (e.g., by global norm)
    grads = optax.clip_by_global_norm(1.0)(grads) # Clip gradients to prevent explosion

    # Update parameters using optimizer
    updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)

    # Calculate metrics (e.g., accuracy)
    predicted_labels = jnp.argmax(logits, axis=-1)
    # Accuracy only on non-padded tokens
    accuracy = jnp.mean((predicted_labels == batch['labels']) * (batch['labels'] != 0))

    # Return updated TrainState and current step metrics
    new_state = state.replace(
        params=new_params,
        opt_state=new_opt_state,
        dropout_rng=new_dropout_rng # Update dropout RNG for next step
    )
    return new_state, loss, accuracy

# --- 4. 模型初始化 ---
def init_model_and_optimizer(rng_key: jax.Array, model_config: dict, learning_rate_schedule: Callable[[int], float]):
    """
    Initializes the Transformer model and Optax optimizer.
    """
    # Split RNG key for parameter initialization and dropout initialization
    init_rng, dropout_init_rng = jax.random.split(rng_key)
    
    # Dummy inputs for shape inference during model initialization
    # These shapes must match what the model expects, batch size can be 1
    dummy_encoder_input = jnp.zeros((1, model_config['max_len']), dtype=jnp.int32)
    dummy_decoder_input = jnp.zeros((1, model_config['max_len']), dtype=jnp.int32)
    
    # Initialize the Transformer model instance
    model = Transformer(
        encoder_num_layers=model_config['num_layers'],
        decoder_num_layers=model_config['num_layers'],
        embed_dim=model_config['embed_dim'],
        num_heads=model_config['num_heads'],
        qkv_features=model_config['embed_dim'], # Typically qkv_features == embed_dim
        mlp_features=model_config['mlp_features'],
        dropout_rate=model_config['dropout_rate'],
        deterministic=True, # Set to True for initialization (no dropout)
        vocab_size=model_config['vocab_size'],
        max_len=model_config['max_len'],
        dtype=model_config.get('dtype', jnp.float32), # Allow specifying dtype
    )
    
    # Generate dummy masks for initialization
    dummy_enc_mask, dummy_dec_self_mask, dummy_cross_mask = create_masks(
        dummy_encoder_input, dummy_decoder_input
    )

    # Initialize model parameters.
    # Pass 'dropout' RNG for any dropout layers, even if deterministic=True during init.
    variables = model.init(
        {'params': init_rng, 'dropout': dropout_init_rng}, 
        dummy_encoder_input, dummy_decoder_input,
        encoder_attention_mask=dummy_enc_mask,
        decoder_self_attention_mask=dummy_dec_self_mask,
        cross_attention_mask=dummy_cross_mask
    )
    params = variables['params']

    # Initialize optimizer (e.g., AdamW)
    optimizer = optax.adamw(learning_rate=learning_rate_schedule)
    opt_state = optimizer.init(params)

    # Create TrainState
    state = TrainState.create(
        apply_fn=model.apply, # Store apply_fn (bound to a model instance)
        params=params,
        tx=optimizer,
        opt_state=opt_state,
        dropout_rng=dropout_init_rng # Store the dropout RNG
    )
    return state

# --- 5. 概念性训练循环 ---

def main():
    # --- Configuration ---
    model_config = {
        'num_layers': 6,
        'embed_dim': 512,
        'num_heads': 8,
        'mlp_features': 2048, # Inner dimension of FFN
        'dropout_rate': 0.1,
        'vocab_size': 10000, # Example vocabulary size
        'max_len': 128,      # Example max sequence length
        'dtype': jnp.float32 # Use jnp.bfloat16 for mixed precision on supported hardware
    }
    training_config = {
        'num_epochs': 10,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'warmup_steps': 1000 # Example for learning rate schedule
    }

    # --- Setup RNG ---
    key = jax.random.PRNGKey(0)
    master_key, init_key, data_key = jax.random.split(key, 3)

    # --- Learning Rate Schedule (example: linear warmup then constant) ---
    def learning_rate_schedule_fn(step):
        return jnp.minimum(step / training_config['warmup_steps'], 1.0) * training_config['learning_rate']

    # --- Initialize Model and Optimizer ---
    print("Initializing model and optimizer...")
    state = init_model_and_optimizer(init_key, model_config, learning_rate_schedule_fn)
    print("Initialization complete.")

    # --- Dummy Data (Replace with your actual DataLoader) ---
    # In a real scenario, you would load data, tokenize it, and create batches.
    # Here, we create dummy batches to show the training loop structure.
    print("Generating dummy data...")
    num_batches_per_epoch = 10
    def generate_dummy_batch(batch_key, batch_size, enc_len, dec_len, vocab_size):
        enc_input = jax.random.randint(batch_key, (batch_size, enc_len), minval=1, maxval=vocab_size)
        dec_input = jax.random.randint(batch_key, (batch_size, dec_len), minval=1, maxval=vocab_size)
        # Shift decoder input to create labels (next token prediction)
        labels = jnp.concatenate([dec_input[:, 1:], jnp.zeros((batch_size, 1), dtype=jnp.int32)], axis=1)
        # Pad some tokens to demonstrate masking (replace some non-zero IDs with 0)
        enc_input = enc_input.at[:, enc_len // 2:].set(0)
        dec_input = dec_input.at[:, dec_len // 2:].set(0)
        labels = labels.at[:, dec_len // 2:].set(0)

        return {
            'encoder_input_ids': enc_input,
            'decoder_input_ids': dec_input,
            'labels': labels,
            'vocab_size': vocab_size # Pass vocab_size for loss function/model init
        }
    print("Dummy data generation complete.")

    # --- Training Loop ---
    print("Starting training loop...")
    for epoch in range(training_config['num_epochs']):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        for i in range(num_batches_per_epoch):
            batch_key, data_key = jax.random.split(data_key) # Split data key for each batch
            batch = generate_dummy_batch(batch_key, training_config['batch_size'], 
                                         model_config['max_len'], model_config['max_len'],
                                         model_config['vocab_size'])
            
            # Get current learning rate from schedule
            current_lr = learning_rate_schedule_fn(state.step)
            
            state, loss, accuracy = train_step(state, batch, current_lr)
            epoch_loss += loss
            epoch_accuracy += accuracy

            if i % 5 == 0:
                print(f"  Epoch {epoch+1}/{training_config['num_epochs']} | Batch {i+1}/{num_batches_per_epoch} | "
                      f"LR: {current_lr:.6f} | Loss: {loss:.4f} | Accuracy: {accuracy:.4f}")
        
        avg_epoch_loss = epoch_loss / num_batches_per_epoch
        avg_epoch_accuracy = epoch_accuracy / num_batches_per_epoch
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_epoch_loss:.4f}, Avg Accuracy: {avg_epoch_accuracy:.4f}\n")

    print("Training finished.")

    # --- Inference Example (Conceptual) ---
    # For actual inference (e.g., greedy decoding, beam search), you would need to:
    # 1. Set deterministic=True for the model apply function.
    # 2. Iteratively feed the decoder output back as input.
    # 3. Handle tokenization/detokenization.
    # Example of a single forward pass for inference:
    print("\n--- Example Inference (Forward Pass) ---")
    inference_key, _ = jax.random.split(master_key)
    dummy_inference_batch = generate_dummy_batch(inference_key, 1, 
                                                 model_config['max_len'], model_config['max_len'],
                                                 model_config['vocab_size'])
    
    enc_mask, dec_self_mask, cross_mask = create_masks(
        dummy_inference_batch['encoder_input_ids'], dummy_inference_batch['decoder_input_ids']
    )

    inference_logits = Transformer(
        encoder_num_layers=model_config['num_layers'],
        decoder_num_layers=model_config['num_layers'],
        embed_dim=model_config['embed_dim'],
        num_heads=model_config['num_heads'],
        qkv_features=model_config['embed_dim'],
        mlp_features=model_config['mlp_features'],
        dropout_rate=model_config['dropout_rate'],
        deterministic=True, # Set to True for inference (no dropout)
        vocab_size=model_config['vocab_size'],
        max_len=model_config['max_len'],
        dtype=model_config.get('dtype', jnp.float32),
    ).apply(
        {'params': state.params}, # Use trained parameters
        dummy_inference_batch['encoder_input_ids'],
        dummy_inference_batch['decoder_input_ids'],
        encoder_attention_mask=enc_mask,
        decoder_self_attention_mask=dec_self_mask,
        cross_attention_mask=cross_mask,
        rngs={'dropout': inference_key} # Still pass RNG, even if deterministic (Flax expects it)
    )
    print("Inference logits shape:", inference_logits.shape)
    print("First few predicted token IDs (conceptual):", jnp.argmax(inference_logits[0, :5], axis=-1))

if __name__ == '__main__':
    main()
