FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/cache
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY migrations ./migrations
COPY deploy/constraints.txt ./deploy/constraints.txt
RUN pip install --no-cache-dir -c deploy/constraints.txt '.[backend,classification]' \
    && useradd --uid 10001 --create-home app \
    && mkdir -p /var/lib/damage/inspections \
    && chown -R app:app /var/lib/damage
USER app
# Compose chooses the backend, severity factory or explicit migration command.
CMD ["python", "-m", "uvicorn", "damage_analytics.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
