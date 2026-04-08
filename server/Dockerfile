FROM python:3.11
WORKDIR /app
COPY . .
RUN pip install fastapi uvicorn pydantic openai "openenv-core[core]>=0.2.1"
CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "7860"]
