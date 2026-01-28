# --------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.13-bookworm

# ---- create a non‑root user -----------------------------------
RUN adduser arena
USER arena
WORKDIR /home/arena

# --------------------------------------------------------------
# Copy source code
COPY pyproject.toml uv.lock ./
COPY src src

# Install dependencies – cache the uv directory for speed
RUN --mount=type=cache,target=/home/arena/.cache/uv,uid=1000 \
    uv sync --locked

# --------------------------------------------------------------
ENTRYPOINT ["uv", "run", "src/server.py"]
CMD ["--host", "0.0.0.0"]
EXPOSE 9009
