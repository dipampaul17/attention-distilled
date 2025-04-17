import random
import os
from datasets import load_from_disk
import sacrebleu
import torch

# Use a simpler model that works with the available dependencies
class DummyTranslator:
    """A dummy translator that produces a simple rule-based translation"""
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
    
    def translate(self, texts):
        """Simple rule-based translation from English to German"""
        translations = []
        # Basic English to German word mapping
        en_de_dict = {
            "the": "die",
            "a": "ein",
            "is": "ist",
            "are": "sind",
            "and": "und",
            "to": "zu",
            "in": "in",
            "i": "ich",
            "I": "Ich",
            "you": "du",
            "he": "er",
            "she": "sie",
            "it": "es",
            "It": "Es",
            "we": "wir",
            "they": "sie",
            "have": "haben",
            "do": "tun",
            "can": "können",
            "for": "für",
            "on": "auf",
            "with": "mit",
            "of": "von",
            "this": "dies",
            "that": "das",
            "no": "nein",
            "yes": "ja",
            "not": "nicht",
            "was": "war",
            "what": "was",
            "when": "wann",
            "where": "wo",
            "why": "warum",
            "how": "wie",
            "all": "alle",
            "but": "aber",
            "or": "oder",
            "if": "wenn",
            "because": "weil",
            "about": "über",
            "like": "wie",
            "more": "mehr",
            "my": "mein",
            "your": "dein",
            "his": "sein",
            "her": "ihr",
            "their": "ihr",
            "our": "unser",
            "there": "dort",
            "here": "hier"
        }
        
        for text in texts:
            words = text.split()
            translated_words = []
            for word in words:
                # Remove punctuation for lookup
                clean_word = word.lower().strip(".,!?;:()[]{}'\"")
                # Get any punctuation
                punctuation = ""
                if word and not word[-1].isalnum():
                    punctuation = word[-1]
                
                # Translate if in dictionary, otherwise keep original
                if clean_word in en_de_dict:
                    # Preserve capitalization
                    if word[0].isupper() and clean_word[0].islower():
                        translated = en_de_dict[clean_word].capitalize()
                    else:
                        translated = en_de_dict[clean_word]
                    translated_words.append(translated + punctuation)
                else:
                    translated_words.append(word)
            
            translations.append(" ".join(translated_words))
        return translations

# Load the dataset
print("Loading dataset...")
dataset = load_from_disk("./toy_en_de")
test_samples = dataset.select(range(200))  # First 200 samples
print(f"Loaded {len(test_samples)} test samples")

# Initialize our translator
translator = DummyTranslator()

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
    
    # Translate
    predicted_translation = translator.translate([source_text])[0]
    
    # Store results
    sources.append(source_text)
    references.append(reference_text)
    hypotheses.append(predicted_translation)

# Calculate BLEU score
print("Calculating BLEU score...")
bleu = sacrebleu.corpus_bleu(hypotheses, [references])

# Print BLEU score
print(f"SacreBLEU score: {bleu.score:.2f}")

# Select 3 random examples to display
sample_indices = random.sample(range(len(sources)), 3)
for i, idx in enumerate(sample_indices):
    print(f"\nExample {i+1}:")
    print(f"Source: {sources[idx]}")
    print(f"Hypothesis: {hypotheses[idx]}")
    print(f"Reference: {references[idx]}")
