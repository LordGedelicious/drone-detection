# Container for training / evaluating / running the from-scratch drone detector.
# Matches the RunPod pod: CUDA 12.8 + cuDNN, Python 3.12, torch 2.8.0+cu128.
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv git libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

WORKDIR /app

COPY requirements-final.txt .
RUN pip install --upgrade pip && pip install -r requirements-final.txt

COPY src/ src/
COPY scripts/ scripts/
COPY splits/ splits/
COPY train.py finetune.py eval.py infer.py ./

# data/, checkpoints/, runs/, wandb/ are bind-mounted at runtime (see compose).
ENTRYPOINT ["python3"]
CMD ["train.py", "--help"]
