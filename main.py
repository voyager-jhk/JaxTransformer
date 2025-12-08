import jax
import jax.numpy as jnp
from src.config import MODEL_CONFIG, TRAINING_CONFIG
from src.train import init_model_and_optimizer, train_step
from src.data import generate_dummy_batch
from src.model import Transformer
from src.utils import create_masks

def main():
    # 1. 随机数种子设置
    key = jax.random.PRNGKey(0)
    master_key, init_key, data_key = jax.random.split(key, 3)

    # 2. 学习率调度器
    def learning_rate_schedule_fn(step):
        return jnp.minimum(step / TRAINING_CONFIG['warmup_steps'], 1.0) * TRAINING_CONFIG['learning_rate']

    # 3. 初始化
    print(f"Initializing model with config: {MODEL_CONFIG}")
    state = init_model_and_optimizer(init_key, MODEL_CONFIG, learning_rate_schedule_fn)
    print("Initialization complete.")

    # 4. 训练循环
    print("Starting training loop...")
    for epoch in range(TRAINING_CONFIG['num_epochs']):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        batches_per_epoch = TRAINING_CONFIG['batches_per_epoch']

        for i in range(batches_per_epoch):
            batch_key, data_key = jax.random.split(data_key)
            batch = generate_dummy_batch(
                batch_key, 
                TRAINING_CONFIG['batch_size'], 
                MODEL_CONFIG['max_len'], 
                MODEL_CONFIG['max_len'],
                MODEL_CONFIG['vocab_size']
            )
            
            current_lr = learning_rate_schedule_fn(state.step)
            state, loss, accuracy = train_step(state, batch, current_lr)
            
            epoch_loss += loss
            epoch_accuracy += accuracy

            if i % 5 == 0:
                print(f"  Epoch {epoch+1} | Batch {i+1} | LR: {current_lr:.6f} | Loss: {loss:.4f} | Acc: {accuracy:.4f}")
        
        avg_loss = epoch_loss / batches_per_epoch
        avg_acc = epoch_accuracy / batches_per_epoch
        print(f"Epoch {epoch+1} Done. Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.4f}\n")

    # 5. 推理演示 (One-shot Forward)
    print("--- Inference Demo ---")
    inference_key, _ = jax.random.split(master_key)
    dummy_batch = generate_dummy_batch(inference_key, 1, MODEL_CONFIG['max_len'], MODEL_CONFIG['max_len'], MODEL_CONFIG['vocab_size'])
    
    enc_mask, dec_self_mask, cross_mask = create_masks(
        dummy_batch['encoder_input_ids'], dummy_batch['decoder_input_ids']
    )

    # 实例化一个推理模式的 Transformer (deterministic=True)
    inference_model = Transformer(
        encoder_num_layers=MODEL_CONFIG['num_layers'],
        decoder_num_layers=MODEL_CONFIG['num_layers'],
        embed_dim=MODEL_CONFIG['embed_dim'],
        num_heads=MODEL_CONFIG['num_heads'],
        qkv_features=MODEL_CONFIG['embed_dim'],
        mlp_features=MODEL_CONFIG['mlp_features'],
        dropout_rate=MODEL_CONFIG['dropout_rate'],
        deterministic=True, # 关键：推理模式
        vocab_size=MODEL_CONFIG['vocab_size'],
        max_len=MODEL_CONFIG['max_len'],
        dtype=MODEL_CONFIG['dtype'],
    )

    logits = inference_model.apply(
        {'params': state.params},
        dummy_batch['encoder_input_ids'],
        dummy_batch['decoder_input_ids'],
        encoder_attention_mask=enc_mask,
        decoder_self_attention_mask=dec_self_mask,
        cross_attention_mask=cross_mask,
        rngs={'dropout': inference_key} 
    )
    
    print(f"Inference logits shape: {logits.shape}")
    print("Predicted tokens (first 5):", jnp.argmax(logits[0, :5], axis=-1))

if __name__ == '__main__':
    main()