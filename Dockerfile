FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock* ./

RUN poetry install --no-dev --no-root

COPY app ./app

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "app.main", "--http", "--workers"]
