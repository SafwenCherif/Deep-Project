FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m appuser \
  && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

ENV CLASSIFIER_PATH=/models/brain_tumor_resnet50_pipeline_f.keras \
    SEGMENTOR_PATH=/models/Brain_Tumor_Segmentation_FTL_Winner.keras

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
