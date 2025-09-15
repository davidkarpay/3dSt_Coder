FROM python:3.12-slim AS builder
WORKDIR /app
COPY poetry.lock pyproject.toml ./
RUN pip install poetry && poetry install --no-root --only main
COPY . .
RUN poetry build

FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /app/dist/*.whl .
RUN pip install *.whl
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]