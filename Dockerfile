# Base image with CUDA support
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# --- training layer ---
RUN pip install --no-cache-dir \
    torch>=2.2 \
    transformers>=4.40 \
    datasets \
    sacrebleu \
    accelerate

# --- inference layer ---
RUN pip install --no-cache-dir fastapi==0.110 uvicorn[standard]==0.29

# Copy required files
COPY serve.py /workspace/
COPY model.pt /workspace/
COPY ckpt /workspace/ckpt/

EXPOSE 8000
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
