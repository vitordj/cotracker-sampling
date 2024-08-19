FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-pip \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home

RUN git clone https://github.com/facebookresearch/co-tracker

RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

WORKDIR /home/co-tracker

RUN pip install -e . \
    && pip install opencv-python einops timm matplotlib moviepy flow_vis

RUN mkir checkpoints \
    cd checkpoints \
    && wget https://huggingface.co/facebook/cotracker/resolve/main/cotracker2.pth

# copia meu repositorio
COPY . .

RUN pip install requiremets.txt

ENTRYPOINT [ "bash" ]