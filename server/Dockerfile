FROM python:3.11
WORKDIR /app
COPY . .
RUN pip install fastapi uvicorn pydantic
CMD ["python", "app.py"]
