FROM python:3.10-slim

# Install system dependencies needed by SciPy, numpy, implicit, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

COPY . /app

ENV DATA_PATH=/app/Data
ENV INFERENCE_API_KEY=dev-key

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
