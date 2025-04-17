from datasets import load_dataset

# Load the dataset
ds = load_dataset("iwslt2017", "iwslt2017-en-de")['train']

# Shuffle and select 1000 samples
ds = ds.shuffle(seed=42).select(range(1000))   # 1k pairs

# Save to disk
ds.save_to_disk("./toy_en_de")
