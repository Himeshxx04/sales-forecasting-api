FROM python:3.11-slim

WORKDIR /app

# system deps needed by Prophet / cmdstan compiler
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# pre-install cmdstan so Prophet works without internet at runtime
RUN python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"

# copy source code
COPY src/ ./src/
COPY api/ ./api/
COPY data/ ./data/
COPY models_saved/ ./models_saved/
COPY train.py .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
