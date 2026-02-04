# %% [markdown]
"""
# Chapter 1 Prerequisite Workbook

This workbook builds the exact skills you need for Chapter 1: Transformer Interpretability.
Each section maps directly to concepts you'll encounter in [1.1] and [1.2].

**How to use this workbook:**
1. Complete the assessment first to identify weak areas
2. Work through sections where you scored < 4
3. Each exercise has a direct connection to Chapter 1 concepts
4. Solutions are provided - use them to check understanding, not skip learning

**Time estimates:**
- Section A (Linear Algebra): 45-60 minutes
- Section B (PyTorch): 45-60 minutes
- Section C (Transformers): 60-90 minutes
- Section D (Interpretability): 30-45 minutes
"""

# %% [markdown]
"""
## Setup
"""

# %%
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import einops
from jaxtyping import Float, Int
from torch import Tensor
import math

MAIN = __name__ == "__main__"

# %%
# Verify your setup
if MAIN:
    print(f"PyTorch version: {t.__version__}")
    print(f"Device: {'cuda' if t.cuda.is_available() else 'mps' if t.backends.mps.is_available() else 'cpu'}")
    print("Setup complete!")

# %% [markdown]
"""
---
# Section A: Linear Algebra for Transformers

These exercises build the exact linear algebra intuitions you need for:
- Understanding attention as matrix multiplication
- Analyzing QK and OV circuits
- Working with low-rank factorizations
- Using SVD for weight analysis

## A1. Matrix Multiplication Shapes

**Chapter 1 Connection:** Every attention computation involves matrix multiplications.
You'll constantly need to track shapes through `Q @ K.T @ V` and similar operations.
"""

# %%
def exercise_a1_shapes():
    """
    Given the following transformer parameters:
    - batch = 4
    - seq_len = 128
    - d_model = 512
    - n_heads = 8
    - d_head = 64 (d_model // n_heads)

    Predict the output shapes for each operation.
    """
    batch, seq_len, d_model, n_heads, d_head = 4, 128, 512, 8, 64

    # Input activations (residual stream)
    residual = t.randn(batch, seq_len, d_model)

    # Weight matrices (one head)
    W_Q = t.randn(d_model, d_head)
    W_K = t.randn(d_model, d_head)
    W_V = t.randn(d_model, d_head)
    W_O = t.randn(d_head, d_model)

    # TODO: Predict each shape before running
    # Exercise: What is the shape of Q = residual @ W_Q?
    Q = residual @ W_Q
    print(f"Q shape: {Q.shape}")  # Your prediction: ___________

    # Exercise: What is the shape of attention_scores = Q @ K.T?
    K = residual @ W_K
    attention_scores = Q @ K.transpose(-2, -1)  # Note: transpose last two dims
    print(f"Attention scores shape: {attention_scores.shape}")  # Your prediction: ___________

    # Exercise: What is the shape after softmax (same shape, why?)
    attention_pattern = F.softmax(attention_scores / math.sqrt(d_head), dim=-1)
    print(f"Attention pattern shape: {attention_pattern.shape}")  # Your prediction: ___________

    # Exercise: What is the shape of V?
    V = residual @ W_V
    print(f"V shape: {V.shape}")  # Your prediction: ___________

    # Exercise: What is attention_out = attention_pattern @ V?
    attention_out = attention_pattern @ V
    print(f"Attention output shape: {attention_out.shape}")  # Your prediction: ___________

    # Exercise: Final output after W_O?
    final_out = attention_out @ W_O
    print(f"Final output shape: {final_out.shape}")  # Your prediction: ___________

if MAIN:
    print("=" * 50)
    print("Exercise A1: Matrix Multiplication Shapes")
    print("=" * 50)
    exercise_a1_shapes()

# %% [markdown]
"""
<details><summary>A1 Solutions & Explanations</summary>

```
Q shape: (4, 128, 64)          # (batch, seq, d_head)
Attention scores: (4, 128, 128) # (batch, seq, seq) - each position attends to all
Attention pattern: (4, 128, 128) # Same - softmax preserves shape
V shape: (4, 128, 64)          # (batch, seq, d_head)
Attention output: (4, 128, 64)  # (batch, seq, d_head) - weighted sum of V
Final output: (4, 128, 512)     # (batch, seq, d_model) - back to residual stream
```

**Key insight for Chapter 1:** The attention pattern has shape (seq, seq) because each
position computes attention to every other position. This is the matrix you'll visualize
when looking for induction heads!

</details>
"""

# %% [markdown]
"""
## A2. Low-Rank Factorizations

**Chapter 1 Connection:** In [1.2], you'll analyze QK and OV circuits as factored matrices.
Understanding low-rank structure is essential for interpretability.
"""

# %%
def exercise_a2_low_rank():
    """
    Explore low-rank structure in attention weight matrices.
    """
    d_model = 256
    d_head = 32  # Much smaller than d_model!

    W_Q = t.randn(d_model, d_head)
    W_K = t.randn(d_model, d_head)

    # The QK circuit: determines attention patterns
    # Full QK matrix would be (d_model, d_model) but we compute it as W_Q @ W_K.T

    # TODO: What is the shape of W_Q @ W_K.T?
    QK_circuit = W_Q @ W_K.T
    print(f"QK circuit shape: {QK_circuit.shape}")

    # TODO: What is the maximum possible rank of QK_circuit?
    # Hint: rank(AB) <= min(rank(A), rank(B))
    rank_bound = min(d_head, d_head)  # Both matrices have rank at most d_head
    print(f"Maximum rank: {rank_bound}")
    print(f"Matrix size: {d_model} x {d_model}")
    print(f"Compression ratio: {d_model * d_model} / {2 * d_model * d_head} = {(d_model * d_model) / (2 * d_model * d_head):.1f}x")

    # Verify with SVD
    U, S, Vh = t.linalg.svd(QK_circuit)
    effective_rank = (S > 1e-6).sum().item()
    print(f"Effective rank (singular values > 1e-6): {effective_rank}")

    # Exercise: Why is this low-rank structure important for interpretability?
    # Think about: if attention patterns are determined by a rank-32 matrix,
    # what does that mean about the "types" of attention patterns possible?

if MAIN:
    print("\n" + "=" * 50)
    print("Exercise A2: Low-Rank Factorizations")
    print("=" * 50)
    exercise_a2_low_rank()

# %% [markdown]
"""
<details><summary>A2 Solutions & Explanations</summary>

```
QK circuit shape: (256, 256)
Maximum rank: 32
Matrix size: 256 x 256
Compression ratio: 8.0x
Effective rank: 32
```

**Key insight for Chapter 1:** The low-rank structure means attention heads can only
implement a limited number of "basis patterns". This is why heads often specialize
(e.g., "attend to previous token" or "attend to same word"). TransformerLens's
`FactoredMatrix` class exploits this for efficient analysis.

</details>
"""

# %% [markdown]
"""
## A3. Einsum Mastery

**Chapter 1 Connection:** TransformerLens and the exercises use einsum/einops extensively.
This is the #1 skill gap that slows people down in Chapter 1.
"""

# %%
def exercise_a3_einsum():
    """
    Master einsum patterns used constantly in transformer code.
    """
    batch, seq, d_model, n_heads, d_head = 2, 8, 64, 4, 16

    # Pattern 1: Batched matrix multiplication
    x = t.randn(batch, seq, d_model)
    W = t.randn(d_model, d_head)

    # These are equivalent:
    result1 = x @ W
    result2 = t.einsum('bsd,dh->bsh', x, W)
    assert t.allclose(result1, result2), "Pattern 1 failed"
    print("Pattern 1 (batched matmul): bsd,dh->bsh ✓")

    # Pattern 2: Attention scores (Q @ K.T)
    Q = t.randn(batch, n_heads, seq, d_head)
    K = t.randn(batch, n_heads, seq, d_head)

    # TODO: Write einsum for Q @ K.T over the last two dimensions
    # Hint: we want (batch, heads, seq_q, seq_k)
    # EXERCISE: Fill in the einsum pattern
    attn_scores_einsum = t.einsum('bhqd,bhkd->bhqk', Q, K)
    attn_scores_manual = Q @ K.transpose(-2, -1)
    assert t.allclose(attn_scores_einsum, attn_scores_manual), "Pattern 2 failed"
    print("Pattern 2 (attention scores): bhqd,bhkd->bhqk ✓")

    # Pattern 3: Weighted sum (attention @ V)
    attn_pattern = F.softmax(attn_scores_einsum / math.sqrt(d_head), dim=-1)
    V = t.randn(batch, n_heads, seq, d_head)

    # TODO: Write einsum for attn_pattern @ V
    # EXERCISE: Fill in the einsum pattern
    weighted_v_einsum = t.einsum('bhqk,bhkd->bhqd', attn_pattern, V)
    weighted_v_manual = attn_pattern @ V
    assert t.allclose(weighted_v_einsum, weighted_v_manual), "Pattern 3 failed"
    print("Pattern 3 (weighted values): bhqk,bhkd->bhqd ✓")

    # Pattern 4: Trace (for composition scores in Chapter 1)
    A = t.randn(d_model, d_model)
    trace_einsum = t.einsum('ii->', A)
    trace_manual = t.trace(A)
    assert t.allclose(trace_einsum, trace_manual), "Pattern 4 failed"
    print("Pattern 4 (trace): ii-> ✓")

    # Pattern 5: Frobenius norm (for composition scores)
    frob_einsum = t.einsum('ij,ij->', A, A).sqrt()
    frob_manual = t.norm(A, 'fro')
    assert t.allclose(frob_einsum, frob_manual), "Pattern 5 failed"
    print("Pattern 5 (frobenius norm): ij,ij-> ✓")

if MAIN:
    print("\n" + "=" * 50)
    print("Exercise A3: Einsum Mastery")
    print("=" * 50)
    exercise_a3_einsum()

# %% [markdown]
"""
<details><summary>A3 Key Patterns Reference</summary>

```python
# Patterns you'll use constantly in Chapter 1:

# Matrix multiply: 'ij,jk->ik'
# Batched matmul: 'bij,jk->bik'
# Attention scores: 'bhqd,bhkd->bhqk'
# Attention output: 'bhqk,bhkd->bhqd'
# Trace: 'ii->'
# Frobenius norm: 'ij,ij->'
# Outer product: 'i,j->ij'
```

**Pro tip:** When confused about einsum, write out the dimension names explicitly
and trace which ones are summed over (appear in inputs but not output).

</details>
"""

# %% [markdown]
"""
## A4. Softmax and Temperature

**Chapter 1 Connection:** Understanding softmax behavior is crucial for analyzing
attention patterns. Temperature scaling appears in sampling and attention analysis.
"""

# %%
def exercise_a4_softmax():
    """
    Develop intuition for softmax behavior.
    """
    # Raw attention scores (before softmax)
    scores = t.tensor([1.0, 2.0, 5.0, 2.0])

    # Standard softmax
    probs_standard = F.softmax(scores, dim=-1)
    print(f"Scores: {scores.tolist()}")
    print(f"Standard softmax: {probs_standard.tolist()}")
    print(f"Max prob: {probs_standard.max():.3f}")

    # TODO: What happens with different temperatures?
    # Temperature > 1: More uniform
    # Temperature < 1: More peaked

    for temp in [0.5, 1.0, 2.0, 10.0]:
        probs = F.softmax(scores / temp, dim=-1)
        print(f"Temp={temp}: max={probs.max():.3f}, entropy={-(probs * probs.log()).sum():.3f}")

    # Exercise: In attention, we divide by sqrt(d_head). Why?
    d_head = 64
    scale = math.sqrt(d_head)
    print(f"\nAttention scale factor for d_head={d_head}: {scale:.2f}")

    # With random scores, the variance grows with d_head
    # This stabilizes the softmax to not be too peaked or too uniform

    # Verify: dot products have variance ~d_head when inputs are unit gaussian
    q = t.randn(1000, d_head)
    k = t.randn(1000, d_head)
    dots = (q * k).sum(dim=-1)  # Element-wise then sum = dot product
    print(f"Dot product std (should be ~sqrt({d_head})={scale:.1f}): {dots.std():.2f}")

if MAIN:
    print("\n" + "=" * 50)
    print("Exercise A4: Softmax and Temperature")
    print("=" * 50)
    exercise_a4_softmax()

# %% [markdown]
"""
---
# Section B: PyTorch for Transformers

These exercises build the PyTorch fluency you need for Chapter 1's exercises.
"""

# %% [markdown]
"""
## B1. Tensor Indexing for Activations

**Chapter 1 Connection:** You'll constantly index into cached activations
to extract specific layers, heads, positions, or tokens.
"""

# %%
def exercise_b1_indexing():
    """
    Practice the indexing patterns used in TransformerLens.
    """
    batch, n_layers, n_heads, seq, d_head = 2, 12, 8, 20, 64

    # Simulated attention patterns cache
    # Shape: (batch, layer, head, query_pos, key_pos)
    attn_patterns = t.randn(batch, n_layers, n_heads, seq, seq)

    # Exercise 1: Get all attention patterns for layer 5
    layer_5 = attn_patterns[:, 5, :, :, :]  # or attn_patterns[:, 5]
    print(f"Layer 5 patterns shape: {layer_5.shape}")
    assert layer_5.shape == (batch, n_heads, seq, seq)

    # Exercise 2: Get attention pattern for layer 3, head 2, all batches
    layer3_head2 = attn_patterns[:, 3, 2, :, :]
    print(f"Layer 3, Head 2 shape: {layer3_head2.shape}")
    assert layer3_head2.shape == (batch, seq, seq)

    # Exercise 3: Get the attention from position 10 to all other positions
    # for all heads in layer 0
    pos10_attention = attn_patterns[:, 0, :, 10, :]
    print(f"Position 10 attention shape: {pos10_attention.shape}")
    assert pos10_attention.shape == (batch, n_heads, seq)

    # Exercise 4: Get attention on the diagonal (self-attention) for layer 0
    # Hint: use torch.diagonal or advanced indexing
    self_attn = t.diagonal(attn_patterns[:, 0], dim1=-2, dim2=-1)
    print(f"Self-attention shape: {self_attn.shape}")

    # Exercise 5: Get the last token's attention pattern (common for next-token prediction)
    last_token_attn = attn_patterns[:, :, :, -1, :]
    print(f"Last token attention shape: {last_token_attn.shape}")
    assert last_token_attn.shape == (batch, n_layers, n_heads, seq)

    print("\nAll indexing exercises passed! ✓")

if MAIN:
    print("\n" + "=" * 50)
    print("Exercise B1: Tensor Indexing")
    print("=" * 50)
    exercise_b1_indexing()

# %% [markdown]
"""
## B2. Einops for Attention

**Chapter 1 Connection:** TransformerLens uses einops extensively for
reshaping between different views of attention heads.
"""

# %%
def exercise_b2_einops():
    """
    Master einops patterns for transformer activations.
    """
    batch, seq, n_heads, d_head, d_model = 2, 10, 8, 32, 256

    # Pattern 1: Split d_model into heads
    # Common when going from residual stream to per-head representations
    x = t.randn(batch, seq, d_model)

    # TODO: Reshape to (batch, seq, n_heads, d_head)
    x_heads = einops.rearrange(x, 'b s (h d) -> b s h d', h=n_heads)
    print(f"Split into heads: {x.shape} -> {x_heads.shape}")
    assert x_heads.shape == (batch, seq, n_heads, d_head)

    # Pattern 2: Reorder for attention computation
    # Attention needs (batch, heads, seq, d_head)
    x_attn = einops.rearrange(x_heads, 'b s h d -> b h s d')
    print(f"Reorder for attention: {x_heads.shape} -> {x_attn.shape}")
    assert x_attn.shape == (batch, n_heads, seq, d_head)

    # Pattern 3: Merge heads back to d_model
    # After attention, go back to residual stream
    x_merged = einops.rearrange(x_attn, 'b h s d -> b s (h d)')
    print(f"Merge heads: {x_attn.shape} -> {x_merged.shape}")
    assert x_merged.shape == (batch, seq, d_model)

    # Pattern 4: Mean over heads (for analysis)
    x_mean_heads = einops.reduce(x_attn, 'b h s d -> b s d', 'mean')
    print(f"Mean over heads: {x_attn.shape} -> {x_mean_heads.shape}")

    # Pattern 5: Repeat for broadcasting
    # Useful when applying same operation to all heads
    scale = t.randn(n_heads)
    scale_broadcast = einops.repeat(scale, 'h -> b h s d', b=batch, s=seq, d=d_head)
    print(f"Broadcast scale: {scale.shape} -> {scale_broadcast.shape}")

    print("\nAll einops exercises passed! ✓")

if MAIN:
    print("\n" + "=" * 50)
    print("Exercise B2: Einops for Attention")
    print("=" * 50)
    exercise_b2_einops()

# %% [markdown]
"""
## B3. Hook Functions

**Chapter 1 Connection:** TransformerLens uses hooks extensively.
Understanding how they work is essential for [1.2] onwards.
"""

# %%
def exercise_b3_hooks():
    """
    Understand how hooks work before using TransformerLens.
    """
    # A simple model to practice hooks on
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(10, 20)
            self.layer2 = nn.Linear(20, 5)

        def forward(self, x):
            x = F.relu(self.layer1(x))
            x = self.layer2(x)
            return x

    model = SimpleNet()

    # Storage for activations
    activations = {}

    # Hook function that stores activations
    def save_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    # Register hooks
    handle1 = model.layer1.register_forward_hook(save_activation('layer1'))
    handle2 = model.layer2.register_forward_hook(save_activation('layer2'))

    # Run forward pass
    x = t.randn(4, 10)
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Layer 1 activation shape: {activations['layer1'].shape}")
    print(f"Layer 2 activation shape: {activations['layer2'].shape}")
    print(f"Output shape: {output.shape}")

    # IMPORTANT: Remove hooks when done
    handle1.remove()
    handle2.remove()

    # Exercise: Verify layer1 activations are ReLU(layer1(x))
    expected_layer1 = F.relu(model.layer1(x))
    assert t.allclose(activations['layer1'], expected_layer1)
    print("\nHook verification passed! ✓")

    # Advanced: Hook that modifies activations
    def ablate_hook(module, input, output):
        """Set all activations to zero (ablation)"""
        return t.zeros_like(output)

    # This is how TransformerLens performs ablations!
    print("\nKey insight: Hooks returning a value REPLACE the activation!")

if MAIN:
    print("\n" + "=" * 50)
    print("Exercise B3: Hook Functions")
    print("=" * 50)
    exercise_b3_hooks()

# %% [markdown]
"""
---
# Section C: Transformer Components

Build each transformer component from scratch to deeply understand it.
"""

# %% [markdown]
"""
## C1. Attention from Scratch

**Chapter 1 Connection:** You'll implement attention in [1.1] and analyze it in [1.2].
"""

# %%
def exercise_c1_attention():
    """
    Implement single-head attention from scratch.
    This is the exact pattern you'll see in Chapter 1.
    """
    batch, seq, d_model, d_head = 2, 8, 64, 16

    # Initialize weights
    W_Q = t.randn(d_model, d_head) / math.sqrt(d_model)
    W_K = t.randn(d_model, d_head) / math.sqrt(d_model)
    W_V = t.randn(d_model, d_head) / math.sqrt(d_model)
    W_O = t.randn(d_head, d_model) / math.sqrt(d_head)

    # Input
    x = t.randn(batch, seq, d_model)

    # TODO: Implement attention step by step

    # Step 1: Compute Q, K, V
    Q = x @ W_Q  # (batch, seq, d_head)
    K = x @ W_K
    V = x @ W_V

    # Step 2: Compute attention scores
    # Scale by sqrt(d_head) for stable gradients
    attn_scores = Q @ K.transpose(-2, -1) / math.sqrt(d_head)  # (batch, seq, seq)

    # Step 3: Apply causal mask (lower triangular)
    # This ensures position i can only attend to positions 0..i
    mask = t.triu(t.ones(seq, seq), diagonal=1).bool()
    attn_scores = attn_scores.masked_fill(mask, float('-inf'))

    # Step 4: Softmax to get attention pattern
    attn_pattern = F.softmax(attn_scores, dim=-1)

    # Step 5: Apply attention to values
    attn_out = attn_pattern @ V  # (batch, seq, d_head)

    # Step 6: Project back to d_model
    output = attn_out @ W_O  # (batch, seq, d_model)

    # Verify shapes
    print(f"Q shape: {Q.shape}")
    print(f"Attention scores shape: {attn_scores.shape}")
    print(f"Attention pattern shape: {attn_pattern.shape}")
    print(f"Attention output shape: {attn_out.shape}")
    print(f"Final output shape: {output.shape}")

    # Check causal mask is working
    print(f"\nCausal mask check (should be lower triangular):")
    print(attn_pattern[0, :4, :4].round(decimals=2))

    # The upper triangle should be 0
    assert (attn_pattern[0].triu(diagonal=1).abs() < 1e-6).all(), "Causal mask failed!"
    print("\nAttention implementation correct! ✓")

if MAIN:
    print("\n" + "=" * 50)
    print("Exercise C1: Attention from Scratch")
    print("=" * 50)
    exercise_c1_attention()

# %% [markdown]
"""
## C2. Multi-Head Attention

**Chapter 1 Connection:** Real transformers use multiple attention heads.
You'll analyze how different heads specialize in [1.2].
"""

# %%
def exercise_c2_multihead():
    """
    Implement multi-head attention, the key innovation of transformers.
    """
    batch, seq, d_model, n_heads = 2, 10, 64, 4
    d_head = d_model // n_heads  # 16

    # Weights: now we have separate weights per head
    # Shape: (n_heads, d_model, d_head)
    W_Q = t.randn(n_heads, d_model, d_head) / math.sqrt(d_model)
    W_K = t.randn(n_heads, d_model, d_head) / math.sqrt(d_model)
    W_V = t.randn(n_heads, d_model, d_head) / math.sqrt(d_model)
    W_O = t.randn(n_heads, d_head, d_model) / math.sqrt(d_head)

    x = t.randn(batch, seq, d_model)

    # Compute Q, K, V for all heads at once using einsum
    Q = t.einsum('bsd,hdk->bhsk', x, W_Q)  # (batch, n_heads, seq, d_head)
    K = t.einsum('bsd,hdk->bhsk', x, W_K)
    V = t.einsum('bsd,hdk->bhsk', x, W_V)

    # Attention scores for all heads
    attn_scores = t.einsum('bhqd,bhkd->bhqk', Q, K) / math.sqrt(d_head)

    # Causal mask
    mask = t.triu(t.ones(seq, seq), diagonal=1).bool()
    attn_scores = attn_scores.masked_fill(mask, float('-inf'))

    # Attention patterns (one per head)
    attn_patterns = F.softmax(attn_scores, dim=-1)

    # Apply attention
    attn_out = t.einsum('bhqk,bhkd->bhqd', attn_patterns, V)

    # Project each head's output and sum
    # W_O: (n_heads, d_head, d_model)
    head_outputs = t.einsum('bhsd,hdm->bhsm', attn_out, W_O)
    output = head_outputs.sum(dim=1)  # Sum over heads

    print(f"Per-head attention patterns shape: {attn_patterns.shape}")
    print(f"Output shape: {output.shape}")

    # Key insight: each head can learn different attention patterns!
    # Let's look at what each head is attending to at position 5
    print(f"\nAttention from position 5 (per head, first batch):")
    for h in range(n_heads):
        top_attended = attn_patterns[0, h, 5, :6].tolist()
        print(f"  Head {h}: {[f'{p:.2f}' for p in top_attended]}")

    print("\nMulti-head attention implemented! ✓")

if MAIN:
    print("\n" + "=" * 50)
    print("Exercise C2: Multi-Head Attention")
    print("=" * 50)
    exercise_c2_multihead()

# %% [markdown]
"""
## C3. Residual Stream and Layer Composition

**Chapter 1 Connection:** Understanding how layers compose through the residual
stream is crucial for mechanistic interpretability.
"""

# %%
def exercise_c3_residual():
    """
    Understand the residual stream as the "memory" of the transformer.
    """
    batch, seq, d_model = 2, 8, 64

    # Initial residual stream (embeddings)
    residual = t.randn(batch, seq, d_model)
    print(f"Initial residual: {residual.shape}")

    # Simulate attention output
    def fake_attention(x):
        return t.randn_like(x) * 0.1  # Small contribution

    # Simulate MLP output
    def fake_mlp(x):
        return t.randn_like(x) * 0.1

    # Layer 0
    attn_out_0 = fake_attention(residual)
    residual = residual + attn_out_0  # Key: ADD, don't replace!

    mlp_out_0 = fake_mlp(residual)
    residual = residual + mlp_out_0

    # Layer 1
    attn_out_1 = fake_attention(residual)
    residual = residual + attn_out_1

    mlp_out_1 = fake_mlp(residual)
    residual = residual + mlp_out_1

    print(f"Final residual: {residual.shape}")

    # Key insight: the final residual is the SUM of all contributions:
    # residual_final = embed + attn_0 + mlp_0 + attn_1 + mlp_1

    # This means we can analyze each component's contribution independently!
    # This is the basis of "logit attribution" in Chapter 1.

    print("\nResidual stream decomposition:")
    print("  final = embedding + sum(attention_outputs) + sum(mlp_outputs)")
    print("  This linearity enables interpretability!")

if MAIN:
    print("\n" + "=" * 50)
    print("Exercise C3: Residual Stream")
    print("=" * 50)
    exercise_c3_residual()

# %% [markdown]
"""
---
# Section D: Interpretability Foundations

These exercises introduce concepts you'll explore deeply in Chapter 1.
"""

# %% [markdown]
"""
## D1. Attention Pattern Analysis

**Chapter 1 Connection:** In [1.2], you'll identify induction heads by their
characteristic attention patterns.
"""

# %%
def exercise_d1_patterns():
    """
    Learn to identify common attention patterns.
    """
    seq = 20

    # Pattern 1: Previous token head
    # Attends to the immediately previous position
    prev_token_pattern = t.zeros(seq, seq)
    for i in range(1, seq):
        prev_token_pattern[i, i-1] = 1.0
    prev_token_pattern[0, 0] = 1.0  # First token attends to itself

    print("Previous Token Pattern (positions 0-5):")
    print(prev_token_pattern[:6, :6])

    # Pattern 2: Induction head pattern
    # If input is "A B ... A", attend to B (position after previous A)
    # Simplified: attend to position that is offset by a fixed amount from a matching position
    # This creates a diagonal stripe pattern on repeated sequences

    # Simulate a repeating sequence: [1,2,3,4,5,1,2,3,4,5]
    tokens = t.tensor([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    seq_len = len(tokens)

    # Induction pattern: position i attends to j if tokens[j-1] == tokens[i-1]
    # (i.e., the token before j matches the token before i)
    induction_pattern = t.zeros(seq_len, seq_len)
    for i in range(1, seq_len):
        for j in range(1, i):  # Can only attend to previous positions
            if tokens[j-1] == tokens[i-1]:  # Previous tokens match
                induction_pattern[i, j] = 1.0

    # Normalize rows
    row_sums = induction_pattern.sum(dim=-1, keepdim=True)
    row_sums = row_sums.clamp(min=1e-6)  # Avoid division by zero
    induction_pattern = induction_pattern / row_sums

    print("\nInduction Pattern for [1,2,3,4,5,1,2,3,4,5]:")
    print("(Position i attends to j where token[j-1] == token[i-1])")
    print(induction_pattern.round(decimals=2))

    # Key observation: induction heads create "diagonal stripes" on repeated sequences
    # Position 6 (second "1") attends strongly to position 1 (after first "1")
    # Position 7 (second "2") attends strongly to position 2 (after first "2")
    # etc.

    print("\n✓ When you see diagonal stripes on repeated random sequences,")
    print("  you've likely found an induction head!")

if MAIN:
    print("\n" + "=" * 50)
    print("Exercise D1: Attention Pattern Analysis")
    print("=" * 50)
    exercise_d1_patterns()

# %% [markdown]
"""
## D2. Logit Attribution

**Chapter 1 Connection:** Logit attribution helps you understand which
components contribute to the model's predictions.
"""

# %%
def exercise_d2_attribution():
    """
    Understand how to attribute predictions to model components.
    """
    # Simplified setup
    vocab_size = 100
    d_model = 64
    seq = 5

    # Unembedding matrix: converts residual stream to logits
    W_U = t.randn(d_model, vocab_size) / math.sqrt(d_model)

    # Residual stream components (in practice, from different layers/heads)
    embed_contribution = t.randn(seq, d_model) * 0.5
    attn_0_contribution = t.randn(seq, d_model) * 0.3
    mlp_0_contribution = t.randn(seq, d_model) * 0.2

    # Total residual stream
    residual = embed_contribution + attn_0_contribution + mlp_0_contribution

    # Final logits
    logits = residual @ W_U  # (seq, vocab_size)

    # Attribution: how much does each component contribute to the logits?
    # Because the unembedding is linear, we can compute each contribution separately!

    embed_logits = embed_contribution @ W_U
    attn_0_logits = attn_0_contribution @ W_U
    mlp_0_logits = mlp_0_contribution @ W_U

    # Verify: contributions sum to total
    assert t.allclose(embed_logits + attn_0_logits + mlp_0_logits, logits)
    print("✓ Logit contributions sum to total logits!")

    # Focus on a specific token prediction
    target_token = 42
    pos = -1  # Last position

    print(f"\nLogit for token {target_token} at final position:")
    print(f"  Total: {logits[pos, target_token]:.3f}")
    print(f"  From embedding: {embed_logits[pos, target_token]:.3f}")
    print(f"  From attention_0: {attn_0_logits[pos, target_token]:.3f}")
    print(f"  From mlp_0: {mlp_0_logits[pos, target_token]:.3f}")

    # Key insight: we can trace back which components "caused" a prediction!
    print("\n✓ This is how we identify which attention heads or MLPs")
    print("  are responsible for specific model behaviors.")

if MAIN:
    print("\n" + "=" * 50)
    print("Exercise D2: Logit Attribution")
    print("=" * 50)
    exercise_d2_attribution()

# %% [markdown]
"""
## D3. Composition Scores

**Chapter 1 Connection:** In [1.2], you'll compute composition scores to
find which heads "talk to" other heads.
"""

# %%
def exercise_d3_composition():
    """
    Understand how attention heads compose with each other.
    """
    d_model = 64
    d_head = 16

    # Head A in layer 0: writes to residual stream via W_O
    W_O_A = t.randn(d_head, d_model) / math.sqrt(d_head)

    # Head B in layer 1: reads from residual stream via W_Q
    W_Q_B = t.randn(d_model, d_head) / math.sqrt(d_model)

    # Composition: does the output of A influence the queries of B?
    # This happens via: residual += A_output, then Q_B = residual @ W_Q_B
    # The A->B path is: A_output @ W_Q_B = ... @ W_O_A @ W_Q_B

    # The "QK composition score" measures this:
    composition_matrix = W_O_A @ W_Q_B  # (d_head, d_head)

    # We summarize this as a single score (Frobenius norm)
    composition_score = t.norm(composition_matrix, 'fro')

    # Baseline: what score do we expect by chance?
    # Random matrices have expected Frobenius norm of sqrt(d_head * d_head) * std
    expected_baseline = math.sqrt(d_head * d_head) * (1/math.sqrt(d_model)) * (1/math.sqrt(d_head))

    print(f"Composition matrix shape: {composition_matrix.shape}")
    print(f"Composition score: {composition_score:.3f}")
    print(f"Expected baseline: {expected_baseline:.3f}")
    print(f"Ratio: {composition_score / expected_baseline:.2f}x baseline")

    # In Chapter 1, you'll find that induction heads have HIGH composition scores
    # with "previous token" heads in earlier layers!

    print("\n✓ High composition scores indicate heads that 'work together'")
    print("  This is how induction circuits are discovered.")

if MAIN:
    print("\n" + "=" * 50)
    print("Exercise D3: Composition Scores")
    print("=" * 50)
    exercise_d3_composition()

# %% [markdown]
"""
---
# Workbook Complete!

## Summary of Key Concepts for Chapter 1

### Linear Algebra
- Matrix multiplication shapes flow: `(batch, seq, d_model) @ (d_model, d_head) -> (batch, seq, d_head)`
- QK circuits are low-rank: factored as `W_Q @ W_K.T`
- Einsum is essential: `'bhqd,bhkd->bhqk'` for attention scores

### PyTorch
- Indexing: `activations[:, layer, head, :, :]`
- Einops: `rearrange(x, 'b s (h d) -> b h s d', h=8)`
- Hooks: `module.register_forward_hook(fn)` to capture activations

### Transformers
- Attention: `softmax(Q @ K.T / sqrt(d_head)) @ V`
- Causal mask: upper triangular set to -inf
- Residual stream: `final = embed + sum(layer_outputs)`

### Interpretability
- Induction heads: diagonal stripes on repeated sequences
- Logit attribution: decompose predictions by component
- Composition scores: measure head-to-head information flow

## Next Steps

1. Re-take the assessment to verify improvement
2. Proceed to [1.1] Transformer from Scratch
3. Keep this workbook handy as a reference

Good luck with Chapter 1!
"""
