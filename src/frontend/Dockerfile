FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY src/frontend/requirements.txt /app/src/frontend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/src/frontend/requirements.txt

COPY src /app/src

EXPOSE 8501

CMD ["streamlit", "run", "src/frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
