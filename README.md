# LLM Course — Building a Transformer LM from Scratch

Welcome to Assignment 1! In this assignment, you will build all of the core components needed to train and run a decoder-only Transformer language model (GPT-style) completely from first principles.

---

## 📖 Interactive Student Guide

For detailed explanations, mathematical formulas, implementation tips, and test mappings for every function, open the included HTML guide in your browser:

👉 **[`cs336_assignment1_student.html`](./cs336_assignment1_student.html)** *(Double-click or open with Chrome/Edge)*

---

## 🚀 Environment Setup

### 1. Create and Activate Conda Environment

> [!IMPORTANT]
> Always make sure your terminal prompt shows `(cs336_basics)` before running any test or script!

```bash
# 1. Create conda environment with Python 3.10
conda create -n cs336_basics python=3.10 -y

# 2. Activate the environment (REQUIRED every time you open a terminal)
conda activate cs336_basics

# 3. Install required packages
pip install -r requirements.txt
```

---

## 🧪 Running Unit Tests

All tests are designed to be run from the root repository directory inside the `cs336_basics` environment:

```bash
# Make sure your conda environment is active
conda activate cs336_basics

# Run specific test suites:
python -m pytest tests/test_train_bpe.py -v
python -m pytest tests/test_tokenizer.py -v
python -m pytest tests/test_model.py -v
python -m pytest tests/test_nn_utils.py -v
python -m pytest tests/test_optimizer.py -v
python -m pytest tests/test_serialization.py -v
python -m pytest tests/test_data.py -v

# Run all tests:
python -m pytest tests/ -v --tb=short
```

---

## 📁 Repository Structure

```text
llm-course/
├── cs336_assignment1_student.html  # Interactive visual guide for students
├── requirements.txt                # Python dependencies
├── tests/
│   ├── adapters.py                 # ✏️ Implementation stubs (functions you fill in)
│   ├── helper_classes.py           # ✏️ Implementation stubs (classes you fill in)
│   ├── helper_functions.py         # ✓ GIVEN: pre-tokenizer regex & split utilities
│   ├── common.py                   # ✓ GIVEN: fixtures path & test utilities
│   ├── fixtures/                   # Reference weights & test tensors
│   ├── test_train_bpe.py           # Tests for BPE tokenizer training
│   ├── test_tokenizer.py           # Tests for Tokenizer class (encode/decode)
│   ├── test_model.py               # Tests for RMSNorm, GELU, Attention, LM
│   ├── test_nn_utils.py            # Tests for Softmax, Cross-Entropy, Gradient Clipping
│   ├── test_optimizer.py           # Tests for AdamW and Cosine LR Scheduler
│   ├── test_serialization.py       # Tests for checkpoint save/load
│   └── test_data.py                # Tests for memory-efficient batch sampling
```

---

## 🛠️ What You Will Implement

### Section 2: BPE Tokenizer
- `MergePriority` in `tests/helper_classes.py` (max-heap tie-breaking)
- `run_train_bpe` in `tests/adapters.py` (byte-pair encoding training)
- `Tokenizer` class in `tests/helper_classes.py` (`encode`, `decode`, `encode_iterable`, `from_files`)

### Section 3: Transformer Architecture
- `RMSNorm` in `tests/helper_classes.py` & `run_rmsnorm` in `tests/adapters.py`
- `run_gelu` in `tests/adapters.py`
- `run_positionwise_feedforward` in `tests/adapters.py`
- `run_softmax` in `tests/adapters.py` (numerically stable)
- `run_scaled_dot_product_attention` in `tests/adapters.py`
- `run_multihead_self_attention` & `CausalMultiHeadSelfAttention` (causal masking, batched QKV)
- `run_transformer_block` (pre-norm residual block)
- `run_transformer_lm` (full language model forward pass with tied embeddings)

### Section 4: Training Components
- `run_cross_entropy` (stable log-sum-exp loss)
- `get_adamw_cls` (AdamW optimizer with decoupled weight decay)
- `run_get_lr_cosine_schedule` (cosine decay with linear warmup)
- `run_gradient_clipping` (global L2 norm clipping)

### Section 5 & 6: Data, Checkpoints & Generation
- `run_get_batch` in `tests/adapters.py` (random sampling from contiguous token stream)
- `run_save_checkpoint` and `run_load_checkpoint`
- Autoregressive text generation with temperature and top-p (nucleus) sampling

---

## 📦 Datasets

To train your models in Section 7:

```bash
mkdir data
cd data

# TinyStories dataset (for quick experimentation)
curl -O https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
curl -O https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# OpenWebText sample
curl -O https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
curl -O https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
# On Windows PowerShell, unpack using tar or gzip:
tar -xzf owt_train.txt.gz
tar -xzf owt_valid.txt.gz

cd ..
```
