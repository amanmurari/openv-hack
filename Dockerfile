FROM python:3.11-slim

LABEL maintainer="OpenEnv Hackathon"
LABEL description="Autonomous Traffic Control – OpenEnv Environment"

WORKDIR /app

# Install dependencies first (layer cache friendly)
COPY pyproject.toml ./
RUN pip install --no-cache-dir \
    "fastapi>=0.104.0" \
    "uvicorn[standard]>=0.24.0" \
    "pydantic>=2.5.0" \
    "requests>=2.31.0" \
    "python-multipart>=0.0.6"

# Copy source
COPY models.py client.py baseline_agent.py inference.py ./
COPY server/ server/

# Install the package itself (editable so imports resolve)
RUN pip install --no-cache-dir -e .

# Runtime config
ENV WORKERS=2
ENV PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["sh", "-c", \
     "uvicorn traffic_control_env.server.app:app \
      --host 0.0.0.0 \
      --port ${PORT} \
      --workers ${WORKERS}"]
