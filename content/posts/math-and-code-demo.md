---
title: "Math & Code Demo"
date: "2026-01-01"
description: "A demonstration of math equations and code blocks in blog posts."
---

This post demonstrates the math and code rendering capabilities of this blog. Let's dive in!

## Inline Math

Einstein's famous equation $E = mc^2$ shows the relationship between energy and mass. In machine learning, we often use the sigmoid function $\sigma(x) = \frac{1}{1 + e^{-x}}$ for activation.

The gradient descent update rule is $\theta := \theta - \alpha \nabla J(\theta)$ where $\alpha$ is the learning rate.

## Block Math

The softmax function is defined as:

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}
$$

The cross-entropy loss function:

$$
L = -\sum_{i=1}^{n} y_i \log(\hat{y}_i)
$$

A more complex example - the attention mechanism in transformers:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

## Inline Code

You can use `console.log()` in JavaScript or `print()` in Python. The `numpy` library is essential for numerical computing, and you might run `pip install torch` to get PyTorch.

## Code Blocks

### Python

```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x
```

### TypeScript

```typescript
interface Model {
    name: string;
    parameters: number;
    accuracy: number;
}

async function trainModel(config: Model): Promise<void> {
    console.log(`Training ${config.name} with ${config.parameters} params`);
    
    const result = await fetch('/api/train', {
        method: 'POST',
        body: JSON.stringify(config),
        headers: { 'Content-Type': 'application/json' }
    });
    
    if (!result.ok) {
        throw new Error('Training failed');
    }
}
```

### Bash

```bash
#!/bin/bash

# Setup Python environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch transformers datasets

# Run training
python train.py --model gpt2 --epochs 10 --lr 1e-4
```

## Combined Example

In neural networks, we compute the forward pass. For a simple linear layer, given input $x$ and weights $W$, the output is:

$$
y = Wx + b
$$

Here's how you'd implement it:

```python
def linear_forward(x, W, b):
    """
    Compute y = Wx + b
    
    Args:
        x: Input tensor of shape (batch_size, in_features)
        W: Weight matrix of shape (out_features, in_features)
        b: Bias vector of shape (out_features,)
    
    Returns:
        Output tensor of shape (batch_size, out_features)
    """
    return x @ W.T + b
```

The gradient with respect to $W$ is $\frac{\partial L}{\partial W} = \frac{\partial L}{\partial y} \cdot x^T$.

## Conclusion

This blog now supports:
- **Inline math** using `$...$` syntax
- **Block math** using `$$...$$` syntax
- **Inline code** using backticks
- **Code blocks** with syntax highlighting for multiple languages
