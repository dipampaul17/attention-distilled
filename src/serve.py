from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = FastAPI(title="Tiny-Transformer EN→DE")

# Load the PyTorch model from saved checkpoint
try:
    # First try to load from ckpt directory
    tokenizer = AutoTokenizer.from_pretrained("./ckpt")
    model = AutoModelForSeq2SeqLM.from_pretrained("./ckpt")
    print("Loaded model from ./ckpt directory")
except Exception as e:
    print(f"Couldn't load from ./ckpt: {e}")
    # Fall back to the saved model.pt file
    print("Loading from model.pt file")
    checkpoint = torch.load("./model.pt", map_location="cpu")
    from transformers import BartForConditionalGeneration, BartConfig
    model_class = eval(checkpoint['class_name'])  # Get the model class from saved checkpoint
    model = model_class(checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    print("Model loaded from model.pt file")

model.eval()  # Set to evaluation mode

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

class Item(BaseModel):
    text: str
    beam: int = 4
    max_len: int = 64

def generate(text, beam=4, max_len=64):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    # Generate translation
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_len,
            num_beams=beam,
            early_stopping=True
        )
    
    # Decode the generated translation
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation

@app.post("/translate")
def translate(item: Item):
    out = generate(item.text, item.beam, item.max_len)
    return {"translation": out}

# Add a simple health check endpoint
@app.get("/health")
def health():
    return {"status": "ok"}

# Add a root endpoint with basic information
@app.get("/")
def root():
    return {
        "service": "English to German Translation API",
        "model": "Fine-tuned BART for EN→DE translation",
        "usage": "Send a POST request to /translate with JSON body containing 'text' field"
    }

# Run with: uvicorn serve:app --host 0.0.0.0 --port 8080
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
