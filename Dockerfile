FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Python runtime settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files first (better layer caching)
COPY pyproject.toml uv.lock ./

# Install production dependencies
RUN uv sync --frozen --no-dev

# Copy application source
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start the application
CMD ["uv", "run", "uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
