# JAX Transformer from Scratch

这是一个使用 JAX 和 Flax 库从零开始实现的 Transformer 模型。该项目旨在提供一个清晰、完整的代码骨架，用于理解和构建序列到序列（Seq2Seq）任务的 Transformer 架构，例如机器翻译。

该实现包含完整的编码器-解码器结构，以及所有核心组件，如多头注意力、位置编码和前馈网络。

## 主要特性

- **完整的 Transformer 架构**：包含一个编码器和一个解码器，适用于序列到序列任务。
- **模块化设计**：所有核心组件（`MultiHeadAttention`、`EncoderBlock`、`DecoderBlock` 等）都被封装为独立的 Flax `nn.Module`，易于理解和扩展。
- **训练流程骨架**：包含一个使用 **Optax** 优化器和 `flax.training.TrainState` 进行训练的示例循环，展示了如何在 JAX 的函数式范式下管理模型状态。
- **JAX 原生实现**：充分利用 JAX 的核心功能，如 **JIT 编译** (`jax.jit`)，以实现高性能计算。

## 项目结构

```
.
├── model.py  # 完整的 Transformer 模型代码和训练流程
├── requirements.txt      # Python 依赖列表
└── README.md             # 项目说明文件
```

## 依赖

本项目需要以下 Python 库。你可以使用 `pip` 轻松安装它们：

```bash
pip install -r requirements.txt
```

文件 `requirements.txt` 内容如下：

```
jax
jaxlib
flax
optax
```

## 如何运行

你可以直接运行 `model.py` 文件来查看模型的初始化过程和一个概念性的训练循环。该脚本包含虚拟数据，可以立即运行。

```bash
python model.py
```

执行该脚本后，你将看到模型初始化、虚拟数据训练的日志，以及一个简单的推理示例。

## 文件详情

### `model.py`

这个文件包含了所有核心代码，从基础的模块定义到完整的 `Transformer` 类。它还展示了：

- **掩码生成**：如何为编码器和解码器创建填充掩码和因果掩码。
- **训练步骤**：如何使用 `jax.value_and_grad` 计算梯度，并通过 **Optax** 更新模型参数。