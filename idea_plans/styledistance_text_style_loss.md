# StyleDistance Projection Loss for Text Generation

Use StyleDistance embeddings as a frozen discriminator to preserve writing style during LLM fine-tuning, analogous to how ArcFace preserves face identity in diffusion training.

## Background

### StyleDistance Model
- **Paper:** "StyleDistance: Stronger Content-Independent Style Embeddings with Synthetic Parallel Examples" (NAACL 2025, arxiv:2410.12757)
- **HuggingFace:** `StyleDistance/styledistance` (English), `StyleDistance/mstyledistance` (multilingual)
- **Architecture:** RoBERTa-base + LoRA, mean pooling → 768-dim embedding
- **License:** MIT
- **Size:** 125M params, ~2ms inference on GPU
- **Key property:** Trained with synthetic contrastive triplets that hold content constant and vary only style. Captures 40 style features across 7 categories (syntactic, lexical, emotional, social, stylistic, graphical, temporal).

### The Analogy

| Diffusion Pipeline | Text Pipeline |
|---|---|
| x0_pred (decoded pixels) | Generated text / soft embeddings |
| ArcFace (frozen, 512-dim) | StyleDistance (frozen, 768-dim) |
| Face identity embedding | Writing style embedding |
| cos_sim vs reference face | cos_sim vs reference style |
| Preserves: who the person is | Preserves: how the author writes |
| Doesn't memorize: pixel positions | Doesn't memorize: specific sentences |

## Design

### Reference Style Embedding (Cached Once)

At startup, compute the target style embedding from reference text:

```python
from sentence_transformers import SentenceTransformer

style_model = SentenceTransformer('StyleDistance/styledistance')
style_model.eval()
for p in style_model.parameters():
    p.requires_grad_(False)

# Compute reference from multiple samples of target author's writing
ref_texts = load_reference_texts()  # list of paragraphs/sentences
ref_embeddings = style_model.encode(ref_texts, convert_to_tensor=True)  # (N, 768)
ref_style = ref_embeddings.mean(dim=0)  # (768,) average style
ref_style = F.normalize(ref_style, dim=-1)
```

Like the ArcFace average embedding — one vector representing the target style.

### Three Integration Paths (Ordered by Practicality)

#### Path A: Hidden State Probe (Cheapest)

Train a lightweight probe to predict StyleDistance embeddings from LLM hidden states, then use the probe as the loss function during fine-tuning.

**Phase 1 — Train the probe:**
```python
# Generate text from the frozen base LLM
# For each generation, extract hidden states AND compute StyleDistance embedding
# Train: hidden_states → Linear(hidden_dim, 768) → MSE vs StyleDistance embedding
```

**Phase 2 — Fine-tune with probe loss:**
```python
# During LLM fine-tuning:
hidden_states = llm(input_ids, output_hidden_states=True).hidden_states[-1]  # (B, seq, hidden)
pooled = hidden_states.mean(dim=1)  # (B, hidden)
pred_style = frozen_probe(pooled)  # (B, 768)
pred_style = F.normalize(pred_style, dim=-1)

style_loss = 1.0 - F.cosine_similarity(pred_style, ref_style.expand_as(pred_style), dim=-1)
total_loss = ce_loss + style_weight * style_loss.mean()
```

**Pros:** Essentially free compute. Direct backprop, no tricks.
**Cons:** Probe accuracy limits the signal quality. Only works for properties linearly decodable from hidden states.
**Cost:** One linear layer forward pass (negligible).

#### Path B: Soft Embedding Path (Most Analogous to TAESD)

Feed soft token representations directly through the frozen StyleDistance encoder.

```python
# LLM produces logits
logits = llm(input_ids).logits  # (B, seq, vocab)

# Soft tokens: differentiable relaxation
soft_probs = F.softmax(logits / tau, dim=-1)  # (B, seq, vocab)
# Map to RoBERTa embedding space
# Option 1: Use StyleDistance's own embedding matrix
roberta_embeds = style_model[0].auto_model.embeddings.word_embeddings.weight  # (vocab_roberta, 768)
# Need vocab alignment between LLM tokenizer and RoBERTa tokenizer — this is the hard part
```

**The tokenizer mismatch problem:** The LLM (e.g., LLaMA) uses a different tokenizer than RoBERTa. Soft probabilities over LLaMA's vocabulary can't directly multiply RoBERTa's embedding matrix.

**Solutions:**
1. **Decode to text, re-encode:** Use Gumbel-Softmax STE to get hard tokens, decode to string, re-tokenize with RoBERTa tokenizer. Gradient flows through STE.
2. **Shared subword mapping:** Build a projection matrix mapping LLM vocab → RoBERTa vocab based on shared subword pieces. Approximate but differentiable.
3. **Bypass tokenizer:** Feed soft embeddings through a small adapter network trained to map LLM embedding space → StyleDistance embedding space (similar to Path A but operating on output embeddings rather than hidden states).

**Pros:** Uses the full StyleDistance model (most accurate).
**Cons:** Tokenizer mismatch is messy. Temperature tuning needed.
**Cost:** StyleDistance forward pass (~2ms) + soft embedding computation.

#### Path C: Gumbel-Softmax STE (GRADE-style, Most Validated)

Use straight-through estimator to get discrete tokens in forward pass, continuous gradients in backward pass.

```python
# LLM produces logits
logits = llm(input_ids).logits  # (B, seq, vocab)

# Gumbel-Softmax STE: hard forward, soft backward
gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
y_soft = F.softmax((logits + gumbel_noise) / tau, dim=-1)
y_hard = F.one_hot(y_soft.argmax(dim=-1), logits.shape[-1]).float()
tokens_ste = (y_hard - y_soft).detach() + y_soft  # hard forward, soft backward

# Decode hard tokens to text
token_ids = tokens_ste.argmax(dim=-1)  # (B, seq)
generated_texts = tokenizer.batch_decode(token_ids, skip_special_tokens=True)

# Run frozen StyleDistance on generated text
with torch.no_grad():
    gen_style = style_model.encode(generated_texts, convert_to_tensor=True)  # (B, 768)
    # But this breaks the gradient chain — StyleDistance.encode doesn't see the STE graph

# ALTERNATIVE: re-embed the STE tokens through StyleDistance's own embedding
roberta_ids = roberta_tokenizer(generated_texts, return_tensors='pt', padding=True)
# ... this still breaks the chain because decode→re-tokenize is non-differentiable
```

**The fundamental issue:** Gumbel-STE only works if the downstream model (StyleDistance) receives the soft/hard tokens *in the same computational graph*. Decoding to string and re-tokenizing breaks the graph.

**Fix:** The GRADE paper handles this by having the reward model share the same tokenizer as the policy, or by operating in embedding space. For cross-tokenizer setups, Path A (probe) or Path B (adapter) are more practical.

### Recommended Path: A (Probe) First, B (Adapter) Later

Path A is the simplest and most likely to work immediately. It's the text equivalent of what we already do with ArcFace — a frozen encoder that evaluates a specific property and provides gradient signal.

If Path A's probe accuracy is insufficient (can't capture complex style features from hidden states alone), upgrade to Path B with a learned adapter network.

## Implementation Plan

### Phase 1: Style Probe Training

**New file:** `toolkit/style_loss.py`

```python
class StyleProbe(nn.Module):
    """Linear probe mapping LLM hidden states to StyleDistance embedding space."""
    def __init__(self, hidden_dim, style_dim=768):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, style_dim)
    
    def forward(self, hidden_states):
        # hidden_states: (B, seq, hidden) or (B, hidden) if pre-pooled
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.mean(dim=1)  # mean pool
        return F.normalize(self.proj(hidden_states), dim=-1)
```

**Training script:** `scripts/train_style_probe.py`
1. Load frozen LLM + frozen StyleDistance model
2. Generate N text samples from the LLM (or use existing corpus)
3. For each sample: extract hidden states from LLM, compute StyleDistance embedding
4. Train the probe (linear regression in embedding space)
5. Save probe weights

### Phase 2: Style Loss Integration

**Config additions:**
```yaml
style_loss:
  enabled: false
  weight: 0.01
  probe_path: "path/to/trained_probe.pt"  # or "auto" to train on-the-fly
  reference_texts: "path/to/author_samples/"  # directory of .txt files
  reference_embedding: null  # or precomputed .pt file
  pool_method: "mean"  # mean, last_token, weighted
```

**Training loop integration:**
```python
if style_loss_config.enabled:
    # Compute probe prediction from LLM hidden states
    hidden = model_output.hidden_states[-1]  # (B, seq, hidden)
    pred_style = self.style_probe(hidden)  # (B, 768)
    
    # Compare against cached reference style
    style_cos = F.cosine_similarity(pred_style, ref_style_expanded, dim=-1)  # (B,)
    style_loss = (1.0 - style_cos).mean()
    
    total_loss = total_loss + style_loss_config.weight * style_loss
```

### Phase 3: Validation

1. Fine-tune a small LLM (e.g., 1-3B) on an author's writing WITH style loss
2. Generate text, compute StyleDistance similarity to author's reference
3. Compare against fine-tuning WITHOUT style loss
4. Verify: style loss should improve style similarity without degrading content quality (perplexity, task performance)

### Phase 4: Clean Similarity Targets (Port from Diffusion)

Same concept as ArcFace clean targets: each training sample has a clean StyleDistance score against the reference. The loss becomes `max(0, 1 - cos_sim / clean_cos)` — a sample that naturally differs in style (e.g., dialogue vs narration) gets a lower target.

```python
# At startup: compute clean_cos for each training text
for sample in dataset:
    clean_style = style_model.encode(sample.text)
    sample.clean_style_cos = cos_sim(clean_style, ref_style).clamp(min=0.1)

# During training:
style_loss = torch.clamp(1.0 - style_cos / clean_style_cos, min=0.0)
```

## Style Features Worth Preserving (by Use Case)

### Author Voice Preservation (Fiction/Non-fiction)
Most valuable features: Sentence complexity, formality, humor, metaphor usage, active/passive voice, contraction frequency, sentiment expression patterns.

### Code Style Preservation (Technical Writing)
Most valuable features: Formality, sentence complexity, present-focused tense, deterministic/certain tone.

### Character Voice in Dialogue
Most valuable features: Contractions, offensive language, self-focused vs you-focused, polite tone, emojis, punctuation patterns.

### Brand Voice
Most valuable features: Formality, positive sentiment, inclusive-focused, humor (or lack thereof), sentence fluency.

## Interaction with Other Losses

- **Cross-entropy (primary):** Style loss is auxiliary. CE handles content correctness; style loss handles how the content is expressed.
- **KL penalty (RLHF regularizer):** Both prevent drift from reference, but KL is distribution-level (token probabilities) while style loss is semantic-level (style embedding similarity). They're complementary.
- **DPO preference loss:** Style loss could replace or augment DPO for style-specific preferences. Instead of human preference pairs, you use the StyleDistance score as an automatic reward.

## Open Questions

1. **Probe accuracy vs full model:** How much signal is lost by using a linear probe instead of running full StyleDistance? Need to measure empirically. If the probe achieves >0.8 correlation with true StyleDistance scores, it's probably sufficient.

2. **Which hidden layer is best?** The last layer captures the most task-specific information, but intermediate layers might better represent style (which is more syntactic/structural). Worth probing multiple layers.

3. **Pooling method:** Mean pooling matches StyleDistance's own method. But for causal LLMs, the last token's hidden state is often more informative. Experiment with both.

4. **Style drift during training:** Does the style embedding remain stable as the LLM changes? The probe was trained on the base model's hidden states — as fine-tuning changes the hidden state distribution, the probe may become less accurate. May need periodic probe re-calibration or online adaptation.

5. **Multi-author training:** Like multi-person identity datasets, could have per-author style references. The per-dataset override system already supports this.

6. **Interaction with quantization:** For 4-bit quantized LLMs, hidden states are computed in reduced precision. The probe should be trained on quantized hidden states if the target model is quantized.

7. **StyleDistance on generated vs natural text:** The model was trained on human-written text. LLM-generated text might have distributional differences that affect embedding quality. Worth testing.
