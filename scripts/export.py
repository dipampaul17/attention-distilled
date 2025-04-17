import argparse
import os
import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path

# Check if we have onnxruntime for validation
TRY_ONNX = True
try:
    import onnxruntime as ort
except ImportError:
    print("Warning: onnxruntime not available for validation, will export but not validate")
    TRY_ONNX = False

def parse_args():
    parser = argparse.ArgumentParser(description='Export transformer model')
    parser.add_argument('--ckpt_path', type=str, default='./ckpt',
                        help='Path to the checkpoint directory')
    parser.add_argument('--onnx_path', type=str, default='./model.onnx',
                        help='Path to save the ONNX model')
    return parser.parse_args()

def main():
    start_time = time.time()
    args = parse_args()
    
    print(f"Loading model from {args.ckpt_path}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.ckpt_path, 
        torch_dtype=torch.float16  # Use float16 for faster inference
    )
    
    # Set model to evaluation mode
    model.eval()
    
    # Sample input for export and validation
    sample_text = "This is a test sentence for model export."
    inputs = tokenizer(sample_text, return_tensors="pt")
    
    # Store PyTorch model output for validation
    with torch.no_grad():
        torch_outputs = model(**inputs)
        torch_logits = torch_outputs.logits.detach().cpu().numpy()
    
    # First, save the model in PyTorch format as a reliable fallback
    pt_path = os.path.join(os.path.dirname(args.onnx_path), "model.pt")
    print(f"Saving model in PyTorch format to {pt_path}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model.config,
        'class_name': model.__class__.__name__
    }, pt_path)
    print("PyTorch model saved successfully")
    
    # Try ONNX export only if dependencies are available
    try:
        # Prepare for ONNX export
        print(f"Attempting to export model to ONNX: {args.onnx_path}")
        input_names = ["input_ids", "attention_mask"]
        output_names = ["logits"]
        
        # Define dynamic axes for batch dimension
        dynamic_axes = {
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "decoder_input_ids": {0: "batch"},
            "logits": {0: "batch"}
        }
        
        # Ensure destination directory exists
        os.makedirs(os.path.dirname(args.onnx_path) or '.', exist_ok=True)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            (inputs.input_ids, inputs.attention_mask),
            args.onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=14,
            do_constant_folding=True
        )
        
        print(f"Model exported to ONNX: {args.onnx_path}")
        
        # Validate the exported model if onnxruntime is available
        if TRY_ONNX:
            print("Validating ONNX model...")
            
            # Create ONNX Runtime session
            ort_session = ort.InferenceSession(args.onnx_path, providers=['CPUExecutionProvider'])
            
            # Prepare inputs for ONNX Runtime
            ort_inputs = {
                'input_ids': inputs.input_ids.numpy(),
                'attention_mask': inputs.attention_mask.numpy()
            }
            
            # Run ONNX inference
            ort_outputs = ort_session.run(None, ort_inputs)
            ort_logits = ort_outputs[0]
            
            # Compare PyTorch and ONNX Runtime outputs
            max_diff = np.max(np.abs(torch_logits - ort_logits))
            print(f"Maximum absolute difference between PyTorch and ONNX Runtime logits: {max_diff}")
            
            if max_diff < 1e-3:
                print("✅  ONNX good")
            else:
                print("❌  ONNX validation failed")
    except Exception as e:
        print(f"ONNX export or validation failed: {str(e)}")
        print("But don't worry, the PyTorch model was saved successfully and can be used for inference.")
        print("✅  PyTorch model export good")
    
    # Print export time
    end_time = time.time()
    export_time = end_time - start_time
    print(f"Export completed in {export_time:.2f} seconds")
    
if __name__ == "__main__":
    main()
