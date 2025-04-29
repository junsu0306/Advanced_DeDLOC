FROM nvcr.io/nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
LABEL maintainer="KAU"
LABEL repository="DeDLOC-BERT-tiny"

WORKDIR /home

# Set en_US.UTF-8 locale by default
RUN apt-get update && apt-get install -y locales curl wget && \
    locale-gen en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

RUN apt-get update && apt-get install -y git build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.11.0-1-Linux-x86_64.sh && \
    bash Miniconda3-py310_23.11.0-1-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-py310_23.11.0-1-Linux-x86_64.sh


ENV PATH="/opt/conda/bin:${PATH}"

RUN conda install python=3.10 pip && \
    pip install --no-cache-dir torch torchvision torchaudio && \
    conda clean --all && rm -rf ~/.cache/pip

COPY . DeDLOC-BERT-tiny/
RUN cd DeDLOC-BERT-tiny && rm -rf ~/.cache/pip

RUN pip install --upgrade pip

RUN pip install git+https://github.com/WKJ-00/hivemind.git@main#egg=hivemind \
    && pip install protobuf==3.20.3 \
    && pip install 'accelerate>=0.26.0' \
    && pip install 'transformers[torch]' \
    && pip install --upgrade "pydantic<2.0" \
    && pip install numpy==1.26.4 --force-reinstall \
    && pip install --upgrade "wandb==0.12.21" \
    && pip install nltk

RUN python -m nltk.downloader punkt \
    && python -m nltk.downloader punkt_tab \
    && python -m nltk.downloader wordnet \
    && python -m nltk.downloader omw-1.4

CMD ["bash"]
