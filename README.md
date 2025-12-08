# JAX Transformer from Scratch

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project is a **from-scratch implementation of the Transformer model** using the **JAX** and **Flax** libraries. It is designed to provide a clear, complete code skeleton for understanding and building the **Sequence-to-Sequence (Seq2Seq)** Transformer architecture, typically used for tasks like machine translation.

The implementation includes the full **Encoder-Decoder structure** and all core components, such as **Multi-Head Attention**, **Positional Encoding**, and the **Feed-Forward Network**.

---

## ✨ Key Features

- **Complete Transformer Architecture**: Includes both an **Encoder** and a **Decoder**, suitable for Sequence-to-Sequence tasks.
- **Modular Design**: All core components (`MultiHeadAttention`, `EncoderBlock`, `DecoderBlock`, etc.) are encapsulated as independent **Flax `nn.Module`s**, making the code easy to understand and extend.
- **Training Loop Skeleton**: Contains an example training loop using the **Optax** optimizer and **`flax.training.TrainState`**, demonstrating state management within JAX's functional paradigm.
- **Native JAX Implementation**: Fully leverages JAX's core functionalities, such as **JIT compilation** (`jax.jit`), to achieve high-performance computation on accelerators (GPUs/TPUs).

---

## 📁 Project Structure

```
.
├── transformer_model.py  # Complete Transformer model code and training flow
├── requirements.txt      # Python dependencies list
└── README.md             # Project documentation file
```

---

## ⚙️ Dependencies

This project requires the following Python libraries. You can easily install them using `pip`:

Bash

```
pip install -r requirements.txt
```

The content of the `requirements.txt` file is as follows:

```
jax
jaxlib
flax
optax
```

---

## ▶️ How to Run

You can directly run the `transformer_model.py` file to see the model's initialization process and a conceptual training loop. The script includes dummy data and is immediately runnable.

Bash

```
python transformer_model.py
```

Upon execution, you will see logs detailing model initialization, the dummy data training steps, and a simple inference example.

---

## 📝 File Details

### `transformer_model.py`

This file contains all the core code, from the basic module definitions to the full **`Transformer`** class. It also demonstrates:

- **Masking Generation**: How to create **padding masks** and **causal masks** for the encoder and decoder.
- **Training Step**: How to use **`jax.value_and_grad`** to compute gradients and update model parameters via **Optax**.

---

## 📜 License

This project is licensed under the **MIT License**.
