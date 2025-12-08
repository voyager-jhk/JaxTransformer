# JAX Transformer from Scratch

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A clean, modular implementation of the **Sequence-to-Sequence Transformer** architecture using **JAX** and **Flax**. This project demonstrates best practices for structuring JAX projects, including functional state management, pure function masking, and hardware-accelerated training loops.

## ✨ Features

- **Full Architecture**: Complete Encoder-Decoder implementation with Multi-Head Attention and Positional Encoding.
- **Modular Design**: Code is split into logical modules (`model`, `train`, `data`) for readability and reusability.
- **JAX Best Practices**:
  - Uses `optax` for optimization.
  - Implements `flax.training.TrainState` for state management.
  - Fully JIT-compiled (`@jax.jit`) training steps and loss functions.
- **Zero-Dependency Logic**: Core masking and loss functions are implemented as pure functions in `utils.py`.

## 📁 Project Structure

```text
.
├── main.py               # Entry point used to run training and inference
├── requirements.txt      # Dependencies
└── src/
    ├── config.py         # Hyperparameters and model configuration
    ├── model.py          # Core Transformer definitions (nn.Module)
    ├── train.py          # Training state, step function, and optimizer setup
    ├── data.py           # Synthetic data generation (easy to replace with real loaders)
    └── utils.py          # Utility functions (Mask generation, Loss calculation)
```

## 🚀 Getting Started

### 1. Installation

Ensure you have Python 3.8+ installed.

```bash
pip install -r requirements.txt
```

### 2. Run Training

Execute the main script to initialize the model and start a training session with synthetic data:

```bash
python main.py
```

You will see logs detailing the initialization, training loss per epoch, and a one-shot inference demonstration at the end.

## 🛠 Configuration

You can adjust model hyperparameters (layers, embedding dimensions, heads) and training settings directly in **src/config.py**:

```python
MODEL_CONFIG = {
    'num_layers': 6,
    'embed_dim': 512,
    # ...
}
```

## 📜 License

This project is licensed under the **MIT License**.
