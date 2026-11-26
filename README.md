---
title: PersonaGen
emoji: 💬
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# PersonaGen: Emotionally Intelligent Chatbot with Feedback Loop

A chatbot that detects how you feel, finds how real people responded to similar
situations, and generates a reply in that tone — regenerating it if it isn't
empathetic enough.

## Features

- **Emotion detection** with a DistilRoBERTa classifier (anger, disgust, fear, joy, neutral, sadness, surprise)
- **Retrieval** over ~47k listener replies from the EmpatheticDialogues dataset (TF-IDF, filtered by detected emotion)
- **Emotion-aware prompting**: persona + tone guideline for the detected emotion + the retrieved reply as a tone example
- **Conversation memory** per browser session (last 5 turns are sent to the generator)
- **Empathy feedback loop**: score the reply and regenerate with a refined prompt if below threshold
- **Generators**: OpenAI chat model (`gpt-4o-mini` by default) when an API key is set, local GPT-2 otherwise
- **React UI** showing emotion, empathy score, retries, and whether memory/retrieval were used

## How it works

```
User message
   │
   ├─► Emotion classifier ──────────────► "sadness" (0.99)
   ├─► Retrieve similar EmpatheticDialogues pair (same emotion preferred)
   ├─► Session memory ──────────────────► previous turns
   ▼
Emotion-aware prompt (persona + tone guideline + retrieved example + history)
   ▼
Generate ─► empathy score ─► below threshold? refine prompt and retry (max 3)
   ▼
Best reply + metrics  ─►  saved to session memory
```

## Run locally

Requires Python 3.10+ and Node 18+.

```bash
# 1. Backend
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
cd backend
python prepare_dataset.py           # downloads EmpatheticDialogues and builds data/empathetic_dialogues.json
                                    # (or: python prepare_dataset.py ../empatheticdialogues.tar.gz)
uvicorn main:app --reload --port 8000

# 2. Frontend (new terminal)
cd frontend
npm install
npm start                           # http://localhost:3000 (API calls are proxied to :8000)
```

API docs: http://localhost:8000/docs

To use OpenAI instead of GPT-2, create `backend/.env`:

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini   # optional
```

### Run with Docker (same as production)

```bash
docker build -t personagen .
docker run -p 7860:7860 personagen                          # GPT-2
docker run -p 7860:7860 -e OPENAI_API_KEY=sk-... personagen # OpenAI
```

Open http://localhost:7860. The image bundles the frontend, dataset and both
models, so it starts in a few seconds and needs no downloads at runtime.

## Deploy

### Render (free) — lite version

`render.yaml` deploys a **lite** build (`Dockerfile.lite`) that skips PyTorch and the
local models and uses OpenAI for both emotion detection and replies. It needs about
200 MB of RAM, so it fits Render's free tier. It requires an OpenAI API key.

1. Sign up at https://render.com with your GitHub account.
2. **New → Blueprint**, connect this repository, pick the branch, click **Apply**.
3. Paste your `OPENAI_API_KEY` when Render asks for it.

The app is live at `https://personagen-xxxx.onrender.com` after the build (~5 min).
Free instances sleep after 15 minutes idle; the first visit after that takes ~30–60 s.
Conversation memory is in RAM, so it resets when the instance sleeps.

### Full version (local models)

The full app is one Docker container (~2 GB image, ~1 GB RAM), so it needs a host
with at least 2 GB of memory. Free tiers with 512 MB (Render free, Railway trial)
are too small.

### Hugging Face Spaces (free, recommended)

1. Create a Space at https://huggingface.co/new-space — SDK: **Docker**, template: **Blank**, hardware: **CPU basic (free)**.
2. Upload the code (from the repo root):
   ```bash
   pip install -U huggingface_hub
   huggingface-cli login
   huggingface-cli upload <your-username>/<space-name> . . --repo-type=space \
     --exclude "*.tar.gz" "*node_modules*" "frontend/build/*" "backend/data/*" ".git/*" "*.env" "*__pycache__*"
   ```
   The excludes keep out your `.env` (API key), `node_modules` and the dataset archive.
3. Optional: in the Space's **Settings → Variables and secrets**, add a secret `OPENAI_API_KEY` for much better replies.

The Space builds the Dockerfile (~5–10 min) and is then live at
`https://<your-username>-<space-name>.hf.space`. The YAML block at the top of
this README is the Space configuration.

### Any other Docker host

Render (Starter plan+), Railway, Fly.io, Google Cloud Run, etc. all work: point
them at this repo's `Dockerfile`, give the instance **≥ 2 GB RAM**, and set
`OPENAI_API_KEY` if you want it. The container listens on `$PORT` (default 7860).

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | – | Use OpenAI for generation (otherwise GPT-2) |
| `LOCAL_MODELS` | `true` | `false` = lite mode: no PyTorch, OpenAI does emotion detection too |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `USE_GPT4` | `true` | Set `false` to force GPT-2 even with a key |
| `LOAD_GPT2` | `false` | Also load GPT-2 when using OpenAI (as a fallback) |
| `PRELOAD_MODELS` | `true` | Load models at startup instead of on first request |
| `CORS_ORIGINS` | `http://localhost:3000,...` | Only needed if the UI is hosted on a different domain |
| `REACT_APP_API_URL` | – | Build-time: API URL if the UI is hosted separately |
| `PORT` | `7860` (Docker) | Port to listen on |

## API

`POST /chat`

```json
{ "message": "I'm stressed about work", "persona": "therapist",
  "empathy_threshold": 0.6, "max_retries": 3, "session_id": "abc123" }
```

Response:

```json
{
  "response": "That sounds really overwhelming...",
  "detected_emotion": "fear",
  "emotion_confidence": 0.93,
  "empathy_score": 0.71,
  "retry_count": 1,
  "persona": "therapist",
  "context_used": true,
  "retrieval_used": true,
  "retrieved_context": "What about it made you feel nervous?",
  "bleu_score": 0.04,
  "generator": "openai:gpt-4o-mini"
}
```

Other endpoints: `GET /personas`, `GET /metrics?session_id=`, `DELETE /memory?session_id=`, `GET /health`.

## Project structure

```
PersonaGen/
├── Dockerfile               # full image: frontend + backend + local models
├── Dockerfile.lite          # lite image for 512 MB hosts (OpenAI only)
├── render.yaml              # Render Blueprint (uses Dockerfile.lite)
├── requirements.txt
├── requirements-lite.txt
├── backend/
│   ├── main.py              # FastAPI app, feedback loop, serves the built frontend
│   ├── models.py            # emotion model, generators, retrieval, prompts
│   ├── utils.py             # empathy scoring, BLEU/diversity, session memory
│   └── prepare_dataset.py   # builds data/empathetic_dialogues.json
└── frontend/                # React app (Create React App)
```

## Limitations

- The empathy score is keyword-based, so it rewards phrases like "I understand"
  and can score genuinely warm replies low (e.g. replies to good news).
- GPT-2 is a small base model; its replies are often off-topic. Use an OpenAI key for a real demo.
- This is a research/portfolio project, **not** a mental-health service. It has no
  crisis detection — don't present it as a substitute for professional help.

## Acknowledgments

- [EmpatheticDialogues](https://github.com/facebookresearch/EmpatheticDialogues) (Rashkin et al., 2019)
- [j-hartmann/emotion-english-distilroberta-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)
- Hugging Face Transformers, FastAPI, React
