
FROM python:3.10

RUN pip install mlflow

EXPOSE 5050

CMD ["mlflow", "server", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "/mlruns", "--host", "0.0.0.0", "--port", "5050"]
