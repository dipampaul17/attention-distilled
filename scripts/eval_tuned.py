import random
import os
from datasets import load_from_disk
import sacrebleu
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Constants
BASELINE_BLEU = 2.58  # The BLEU score from zero_shot.py
CHECKPOINT_PATH = "./ckpt"  # Path to the fine-tuned model

class FineTunedTranslator:
    """A translator that uses the fine-tuned model"""
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model from checkpoint
        print(f"Loading fine-tuned model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        print("Model loaded successfully")
    
    def translate(self, texts):
        """Translate a list of English texts to German"""
        translations = []
        
        # Process in batches to avoid memory issues
        batch_size = 8
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize inputs
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, 
                                   max_length=64, return_tensors="pt").to(self.device)
            
            # Generate translations
            outputs = self.model.generate(
                **inputs,
                max_length=64,
                num_beams=4,
                early_stopping=True
            )
            
            # Decode the outputs
            batch_translations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            translations.extend(batch_translations)
        
        return translations

# Load the dataset
print("Loading dataset...")
dataset = load_from_disk("./toy_en_de")
test_samples = dataset.select(range(200))  # First 200 samples
print(f"Loaded {len(test_samples)} test samples")

# Initialize our fine-tuned translator
translator = FineTunedTranslator(CHECKPOINT_PATH)

# Generate translations
references = []
hypotheses = []
sources = []

print("Translating...")
# Process all samples
for i, sample in enumerate(test_samples):
    if i % 20 == 0:
        print(f"Processing sample {i}/200")
        
    # Extract source and reference texts
    source_text = sample['translation']['en']
    reference_text = sample['translation']['de']
    
    # Store source and reference
    sources.append(source_text)
    references.append(reference_text)

# Batch translate all sources at once for efficiency
print("Generating translations...")
hypotheses = translator.translate(sources)

# Calculate BLEU score
print("Calculating BLEU score...")
bleu = sacrebleu.corpus_bleu(hypotheses, [references])

# Print BLEU scores and improvement
print(f"Baseline SacreBLEU score: {BASELINE_BLEU:.2f}")
print(f"Fine-tuned SacreBLEU score: {bleu.score:.2f}")
print(f"Improvement: +{bleu.score - BASELINE_BLEU:.2f} BLEU points")

# Select 3 random examples to display
sample_indices = random.sample(range(len(sources)), 3)
for i, idx in enumerate(sample_indices):
    print(f"\nExample {i+1}:")
    print(f"Source: {sources[idx]}")
    print(f"Hypothesis: {hypotheses[idx]}")
    print(f"Reference: {references[idx]}")
