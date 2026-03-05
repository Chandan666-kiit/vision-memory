# Deploy Vision & Memory (with FAISS persistence)

## Option A: Docker Compose (local or your server)

1. Set your OpenAI key:
   ```bash
   export OPENAI_API_KEY=sk-your-key-here
   ```
2. Run:
   ```bash
   docker-compose up -d
   ```
3. Open http://localhost:8501
4. FAISS memory is stored in Docker volume `faiss_data` and persists across restarts.

## Option B: Render.com (cloud, FAISS on persistent disk)

1. Push this repo to GitHub.
2. Go to [dashboard.render.com](https://dashboard.render.com) → New → Blueprint.
3. Connect the repo; Render will use `render.yaml`.
4. In the new Web Service, add environment variable:
   - `OPENAI_API_KEY` = your OpenAI API key.
5. Ensure a **Persistent Disk** is attached (mount path `/data`, size 1 GB). The blueprint sets `FAISS_INDEX_DIR=/data/faiss_index` so the app uses this disk.
6. Deploy. Your app URL will be like `https://vision-memory-xxxx.onrender.com`.

## Option C: Any Docker host (VPS, Cloud Run, etc.)

1. Build and run with a volume for FAISS:
   ```bash
   docker build -t vision-memory .
   docker run -p 8501:8501 \
     -e OPENAI_API_KEY=sk-your-key \
     -e FAISS_INDEX_DIR=/data/faiss_index \
     -v faiss_data:/data/faiss_index \
     vision-memory
   ```
2. FAISS index is stored in the `faiss_data` volume.

## Env vars

| Variable           | Required | Description |
|--------------------|----------|-------------|
| `OPENAI_API_KEY`   | Yes      | OpenAI API key. |
| `FAISS_INDEX_DIR`  | No       | Directory for FAISS index. Default: `./faiss_index` next to `main.py`. Set to a persistent path in production (e.g. `/data/faiss_index`). |
