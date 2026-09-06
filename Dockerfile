# Hosted demo image (Hugging Face Spaces / any container host).
# Serves the lightweight webapp: Nemotron on Nebius + tools + memory + Tavily.
# Voice runs in the browser (Web Speech API), so no torch/audio deps here.
FROM python:3.11-slim

WORKDIR /app

# Only the light web deps — keeps the image small and the cold start fast.
COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

# App code + the reusable core it imports (src/, utils/, automation/, core/,
# config/). The heavy voice modules are present but never imported by the webapp.
COPY . .

# HF Spaces routes to port 7860.
ENV PORT=7860
EXPOSE 7860

# data/ (SQLite notes/facts) must be writable; HF Spaces runs as a non-root uid.
RUN mkdir -p data && chmod -R 777 data

CMD ["uvicorn", "webapp.app:app", "--host", "0.0.0.0", "--port", "7860"]
