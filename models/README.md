# Model Information

This directory contains saved models and checkpoints for the English-to-German translation system.

## Model Architecture

The default model used in this repository is a fine-tuned version of BART-base:

- **Base Model**: BART-base (facebook/bart-base)
- **Type**: Transformer-based sequence-to-sequence model
- **Parameters**: ~140M parameters
- **Fine-tuned on**: IWSLT2017 English-German dataset (1,000 examples)
- **Training Time**: ~2.5 minutes on a consumer GPU
- **Performance**: Achieves 7.13 BLEU score (+4.55 improvement over baseline)

## Checkpoint Files

After training, the following files will be generated:

- `model.pt`: The PyTorch saved model (complete model state)
- `checkpoint/`: Directory containing training checkpoints
- `checkpoint/checkpoint-750/`: Best checkpoint saved during training

## Model Loading

The model can be loaded in two ways:

1. **From PyTorch format**:
```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./models/checkpoint")
model = AutoModelForSeq2SeqLM.from_pretrained("./models/checkpoint")
```

2. **From saved PT file**:
```python
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")
model.load_state_dict(torch.load("./models/model.pt"))
```

## Using Custom Models

To use your own pre-trained model:

1. Edit the `scripts/finetune.py` script to use a different base model
2. Update the tokenizer and model loading in `src/serve.py`
3. Ensure compatibility with the rest of the pipeline

## Model Performance Metrics

- **Baseline (rule-based translation)**: 2.58 BLEU
- **Fine-tuned model**: 7.13 BLEU
- **Improvement**: +4.55 BLEU points

Professional NMT systems with larger models can achieve 20-22 BLEU on this dataset.
