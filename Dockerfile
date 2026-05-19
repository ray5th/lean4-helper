FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

RUN apt-get update && apt-get install -y \
    python3.11 python3-pip python3.11-dev \
    curl git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install elan (Lean version manager) and the stable Lean 4 toolchain
RUN curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
    | sh -s -- -y --no-modify-path --default-toolchain leanprover/lean4:stable
ENV PATH="/root/.elan/bin:$PATH"

# Confirm lean is available
RUN lean --version

WORKDIR /app

# Install Python deps first (layer-cached separately from source)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860
ENV GRADIO_SERVER_NAME=0.0.0.0

CMD ["python3", "app.py"]
