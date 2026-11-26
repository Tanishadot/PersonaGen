# ---- 1. Build the React frontend ----
FROM node:20-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json ./
RUN npm install --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---- 2. Python backend that also serves the built frontend ----
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hf_cache \
    PORT=7860
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
# Download + build the retrieval dataset and bake both models into the image,
# so the container starts in seconds and needs no downloads at runtime.
RUN cd backend && python prepare_dataset.py && rm data/empatheticdialogues.tar.gz \
 && python -c "from transformers import pipeline; pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base'); pipeline('text-generation', model='gpt2')"

COPY --from=frontend /app/frontend/build ./frontend/build

# Models are baked in above; don't phone home to Hugging Face on every start.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Run as a non-root user (required by Hugging Face Spaces, good practice elsewhere)
RUN useradd -m -u 1000 user && chown -R user /app
USER user

WORKDIR /app/backend
EXPOSE 7860
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
