import jax
import jax.numpy as jnp
import optax
from flax.training import train_state, common_utils
from typing import Callable, Dict

from .model import Transformer
from .utils import create_masks

class TrainState(train_state.TrainState):
    dropout_rng: jax.Array

@jax.jit
def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray, label_smoothing: float = 0.0):
    num_classes = logits.shape[-1]
    one_hot_labels = common_utils.onehot(labels, num_classes=num_classes)
    
    if label_smoothing > 0.0:
        smooth_labels = one_hot_labels * (1.0 - label_smoothing) + label_smoothing / num_classes
        loss = -jnp.sum(smooth_labels * jax.nn.log_softmax(logits), axis=-1)
    else:
        loss = -jnp.sum(one_hot_labels * jax.nn.log_softmax(logits), axis=-1)
    
    padding_mask = (labels != 0).astype(jnp.float32)
    loss = loss * padding_mask
    return jnp.sum(loss) / jnp.sum(padding_mask)

@jax.jit
def train_step(state: TrainState, batch: Dict, learning_rate: float):
    dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

    def loss_fn(params):
        enc_mask, dec_self_mask, cross_mask = create_masks(
            batch['encoder_input_ids'], batch['decoder_input_ids']
        )
        
        # 重建模型以应用当前参数 (利用 state.apply_fn 绑定的配置)
        # 这里动态读取 bound 方法中的配置
        config = state.apply_fn.keywords # 获取初始化时传入的参数
        
        logits = Transformer(
            encoder_num_layers=config['encoder_num_layers'],
            decoder_num_layers=config['decoder_num_layers'],
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads'],
            qkv_features=config['qkv_features'],
            mlp_features=config['mlp_features'],
            dropout_rate=config['dropout_rate'],
            deterministic=False, # 训练时开启 Dropout
            vocab_size=config['vocab_size'],
            max_len=config['max_len'],
            dtype=config['dtype'],
        ).apply(
            {'params': params},
            batch['encoder_input_ids'],
            batch['decoder_input_ids'],
            encoder_attention_mask=enc_mask,
            decoder_self_attention_mask=dec_self_mask,
            cross_attention_mask=cross_mask,
            rngs={'dropout': dropout_rng}
        )
        return cross_entropy_loss(logits, batch['labels']), logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    grads = optax.clip_by_global_norm(1.0)(grads)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)

    predicted_labels = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean((predicted_labels == batch['labels']) * (batch['labels'] != 0))

    new_state = state.replace(
        params=new_params,
        opt_state=new_opt_state,
        dropout_rng=new_dropout_rng
    )
    return new_state, loss, accuracy

def init_model_and_optimizer(rng_key: jax.Array, model_config: Dict, learning_rate_schedule: Callable):
    init_rng, dropout_init_rng = jax.random.split(rng_key)
    
    dummy_encoder_input = jnp.zeros((1, model_config['max_len']), dtype=jnp.int32)
    dummy_decoder_input = jnp.zeros((1, model_config['max_len']), dtype=jnp.int32)
    
    model = Transformer(
        encoder_num_layers=model_config['num_layers'],
        decoder_num_layers=model_config['num_layers'],
        embed_dim=model_config['embed_dim'],
        num_heads=model_config['num_heads'],
        qkv_features=model_config['embed_dim'],
        mlp_features=model_config['mlp_features'],
        dropout_rate=model_config['dropout_rate'],
        deterministic=True, # 初始化时关闭 Dropout
        vocab_size=model_config['vocab_size'],
        max_len=model_config['max_len'],
        dtype=model_config.get('dtype', jnp.float32),
    )
    
    dummy_enc_mask, dummy_dec_self_mask, dummy_cross_mask = create_masks(
        dummy_encoder_input, dummy_decoder_input
    )

    variables = model.init(
        {'params': init_rng, 'dropout': dropout_init_rng}, 
        dummy_encoder_input, dummy_decoder_input,
        encoder_attention_mask=dummy_enc_mask,
        decoder_self_attention_mask=dummy_dec_self_mask,
        cross_attention_mask=dummy_cross_mask
    )
    
    optimizer = optax.adamw(learning_rate=learning_rate_schedule)
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizer,
        opt_state=optimizer.init(variables['params']),
        dropout_rng=dropout_init_rng
    )
    return state