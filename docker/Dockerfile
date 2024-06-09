FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

RUN apt update && \
    apt upgrade -y && \
    apt install -y \
    software-properties-common \
    git git-lfs curl wget \
    vim

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
RUN apt-get install -y tzdata
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt install python-is-python3 -y

# Install conda
# Referred https://fabiorosado.dev/blog/install-conda-in-docker/
ENV CONDA_DIR /opt/conda

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda --version
RUN conda init bash

# Replace version solver of conda with mamba
# RUN conda install -n base conda-libmamba-solver
# RUN conda config --set solver libmamba

# Create environment
RUN conda create -n ffjako -c conda-forge montreal-forced-aligner python=3.10
SHELL ["conda", "run", "-n", "ffjako", "/bin/bash", "-c"]

# Install mamba
# RUN conda install mamba -c conda-forge

RUN conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
RUN pip install jupyter ipywidgets ipykernel nbconvert \
    transformers datasets tokenizers accelerate fugashi unidic-lite sentencepiece \
    omegaconf loguru wandb

# GID of docker
ARG GID=999
RUN groupadd -g $GID docker
RUN usermod -aG docker root

RUN apt install -y mc

RUN mkdir /workspace
WORKDIR /workspace
RUN git clone https://github.com/sappho192/ffxiv-ja-ko-translator.git /workspace

# Set color prompt
ENV TERM=xterm-256color
RUN echo "PS1='\e[92m\u\e[0m@\e[94m\h\e[0m:\e[35m\w\e[0m# '" >> /root/.bashrc

# Set up vim, basically enable the line number, and color scheme to sublime-monokai
RUN git clone https://github.com/sappho192/dotfiles.git /tmp/dotfiles
RUN cp /tmp/dotfiles/.vimrc /root/.vimrc
RUN cp -r /tmp/dotfiles/.vim /root/.vim

ENV HF_ROOT=/huggingface
ENV HF_HOME=/huggingface/misc
ENV HF_DATASETS_CACHE=/huggingface/datasets
ENV TRANSFORMERS_CACHE=/huggingface/transformers