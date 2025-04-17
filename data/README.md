# Dataset Information

This directory contains translation datasets used for training and evaluating the English-to-German translation model.

## Default Dataset

By default, this repository uses the IWSLT2017 English-German dataset, which is automatically downloaded and processed by the `scripts/dl_data.py` script. The dataset specifications:

- **Source**: IWSLT2017 Conference
- **Languages**: English (source) → German (target)
- **Size**: 1,000 examples (reduced from full dataset)
- **Split**: Training set only
- **Format**: Hugging Face Datasets format

## Sample Data

Below are some examples from the dataset:

| English | German |
|---------|--------|
| "I would like to talk to you about whether or not we should put buildings in the way of a hurricane." | "Ich möchte mit Ihnen darüber reden, ob wir Gebäude in den Weg eines Hurrikans stellen sollten oder nicht." |
| "This is not just a problem for Miami." | "Das ist nicht nur ein Problem für Miami." |
| "It's a problem for millions and millions of people who choose to live along coastlines." | "Es ist ein Problem für Millionen und Abermillionen von Menschen, die sich entscheiden, an Küsten zu leben." |

## Using Custom Datasets

To use your own dataset, follow these steps:

1. Prepare your data in the format: one sentence per line, with parallel files for source and target languages
2. Modify the `scripts/dl_data.py` script to load your dataset instead of IWSLT2017
3. Run the modified script to process and save your dataset

## Data Preprocessing

The default preprocessing steps include:
- Tokenization using the BART tokenizer
- Shuffling with seed 42
- Limiting to 1,000 examples for quick training

To modify these preprocessing steps, edit the `scripts/dl_data.py` script.
