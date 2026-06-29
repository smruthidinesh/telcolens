# Deploy TelcoLens to HuggingFace Spaces (with local models)

HuggingFace Spaces gives ~16 GB RAM on the free CPU tier — enough to run the
**local open-source models** (`all-MiniLM-L6-v2` embeddings + `ms-marco-MiniLM`
cross-encoder), so retrieval and reranking are **fully self-hosted, with no
external API and no rate limits**. Generation still uses Groq if you set a key,
otherwise the extractive fallback.

This is the recommended home for the demo (more stable than free Render, and the
natural place for an ML project).

---

## 1. Create the Space
1. huggingface.co → your profile → **New Space**.
2. **Space SDK: Docker** (blank template). Name it e.g. `telcolens`. Hardware: **CPU basic (free)**.

## 2. Add the code
The Space is a git repo. Easiest path — clone it and copy this project in, with the
HF Dockerfile as `Dockerfile`:

```bash
git clone https://huggingface.co/spaces/<your-username>/telcolens hf-space
cd hf-space
# copy the app + sample data + requirements from this repo
cp -r ../telcolens-engine/app ../telcolens-engine/data ../telcolens-engine/requirements*.txt .
cp ../telcolens-engine/Dockerfile.hf ./Dockerfile      # HF uses a file named exactly "Dockerfile"
```

## 3. Add the Space README (with the required frontmatter)
Create `README.md` in the Space starting with this YAML block:

```yaml
---
title: TelcoLens
emoji: 📑
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---
```

## 4. Set secrets (Space → Settings → Variables and secrets)
- `GROQ_API_KEY` = your Groq key — **optional**, enables fluent LLM answers (free tier). Without it, answers are extractive.
- `TELCOLENS_LOCAL_MODELS=1` is already baked into `Dockerfile.hf`, so local embeddings + reranker are on by default. (You do **not** need a Cohere or OpenAI key — that's the point.)

## 5. Push
```bash
git add -A && git commit -m "TelcoLens on HF Spaces (local models)"
git push                      # prompts for your HF username + an access token (Settings → Access Tokens, write scope)
```
HF builds the Docker image (downloads + bakes the two models — a few minutes the
first time), then the Space goes live at `https://<username>-telcolens.hf.space`.

---

## Why this fixes the pain you hit on Render
- **No rate limits on retrieval/rerank** — they run locally, so the "context does
  not specify…" / slow-response problems from the Cohere/Groq free tiers are gone.
- **More RAM/CPU** — the health-check flapping from the 512 MB Render box goes away.
- **Truly semantic retrieval** — MiniLM embeddings mean "profitability" matches an
  "operating margin" passage without any API.

## Notes
- The in-memory index is still per-session (re-upload after a restart). For full
  persistence, the production path is a vector DB (pgvector/Qdrant/OpenSearch) — see
  the roadmap in `docs/PROJECT_GUIDE.md`.
- First request after the Space sleeps (48 h idle) wakes it in a few seconds.
