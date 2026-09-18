FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

# torch CPU-only vient d'un index dédié, pas de PyPI classique
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=7860
EXPOSE 7860

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "100", "run:app"]