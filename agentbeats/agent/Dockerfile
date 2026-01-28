# --------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.13-bookworm

# ---- create a non‑root user -----------------------------------
RUN adduser agent
USER agent
WORKDIR /home/agent

# --------------------------------------------------------------
# Copy source code
COPY pyproject.toml uv.lock ./
COPY src ./src

# Install dependencies – cache the uv directory for speed
RUN --mount=type=cache,target=/home/agent/.cache/uv,uid=1000 \
    uv sync --locked

# --------------------------------------------------------------
ENTRYPOINT ["uv", "run", "src/server.py"]
CMD ["--host", "0.0.0.0"]
EXPOSE 9019
