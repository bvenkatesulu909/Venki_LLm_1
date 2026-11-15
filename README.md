# Venki_llm_1 — A GPT-Style Language Model Built From Scratch

A 2.54M-parameter decoder-only Transformer implemented by hand in PyTorch, without `nn.Transformer`. The repo covers the full lifecycle: a custom BPE tokenizer, pre-training on public-domain books, fine-tuning, quantization, export to GGUF, and serving locally with Ollama.

**Stack:** Python · PyTorch · byte-level BPE · GGUF · Ollama

## What's implemented

- **Tokenizer** (`bpe.py`): GPT-2-style byte-level BPE trained on the corpus
- **Model** (`model.py`): token and learned positional embeddings, pre-LayerNorm residual blocks, explicit causal multi-head self-attention, GELU MLP, weight-tied output head, scaled initialization
- **Data pipeline** (`preprocess.py`): Project Gutenberg books in `data/`
- **Training** (`train.py`) and **fine-tuning** (`finetune.py`)
- **Text generation** (`generate.py`) with temperature sampling
- **Quantization** (`quantize.py`, `export_2bit.py`) down to 2-bit weights
- **Export** to GGUF (`to_gguf.py`) and an Ollama `Modelfile` for local inference
- **Reference checks** (`*_ref.py`): layer-by-layer numerical verification of attention, blocks, embeddings and the forward pass
- **Course material:** `PyTorch_StepByStep_Course.pdf` and `docs.html` explain every step

## Quick start

```bash
pip install torch numpy
python preprocess.py        # build the training corpus
python train.py             # pre-train
python generate.py          # sample text
python to_gguf.py           # export the model
ollama create venki_llm_1 -f Modelfile
ollama run venki_llm_1
```
