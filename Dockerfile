# ── Stage 1: build / install deps ──────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build tools needed by some wheels
RUN pip install --upgrade pip

# Copy only the manifest first → better layer caching
# (deps layer rebuilds only when pyproject.toml changes)
COPY pyproject.toml README.md ./

# Install all dependencies into an isolated prefix
RUN pip install --prefix=/install .

# ── Stage 2: runtime image ──────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY src/ ./src/
COPY main.py ./

# Run as non-root for security
RUN useradd --no-create-home --shell /bin/false appuser
USER appuser

# Env vars are injected at runtime (docker run -e / --env-file)
# Do NOT bake .env into the image

CMD ["python", "main.py"]
