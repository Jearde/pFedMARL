# Set Versions to use
ARG UBUNTU_VERSION=22.04
ARG CUDA_VERSION=12.2.2
ARG PYTHON_VERSION=3.12

ARG BASE_CONTAINER=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

FROM $BASE_CONTAINER

LABEL maintainer="pFedMARL: Personalized Federated Learning with Multi-Agent Off-Policy Reinforcement Learning"

# environment settings
ENV DEBIAN_FRONTEND=noninteractive
ARG USERNAME=python-user
ARG UID=1000
ARG GID=1000
ENV USERNAME=${USERNAME}

# To use the default value of an ARG declared before the first FROM,
# use an ARG instruction without a value inside of a build stage:
ARG CUDA_VERSION
ARG PYTHON_VERSION

# Expose ports
EXPOSE 22 6007 8888

RUN echo "**** Installing apt packages ****"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    git-lfs \
    curl \
    ca-certificates \
    sudo \
    locales \
    openssh-server \
    openssh-client \
    wget \
    unzip \
    htop \
    nvtop \
    nmon \
    net-tools \
    tmux \ 
    software-properties-common \
    libsndfile1-dev \
    sox \
    libsox-dev \
    apt-transport-https \
    gpg \
    libopenblas-openmp-dev \
    iftop \
    speedometer \
    ffmpeg \
    inkscape \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Install kubectl
RUN sudo mkdir -p -m 755 /etc/apt/keyrings
RUN curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.29/deb/Release.key | sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
RUN sudo chmod 644 /etc/apt/keyrings/kubernetes-apt-keyring.gpg # allow unprivileged APT programs to read this keyring
RUN echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.29/deb/ /' | sudo tee /etc/apt/sources.list.d/kubernetes.list
RUN sudo chmod 644 /etc/apt/sources.list.d/kubernetes.list # helps tools such as command-not-found to work correctly
RUN sudo apt-get update && sudo apt-get install -y kubectl


# Use bash instead of sh
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

RUN echo "**** Setting timezone ****"

# Make the "en_US.UTF-8" locale
RUN localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8
ENV LANG=en_US.utf8

# Setup timezone
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Fix for too many open files
RUN sudo sysctl -w fs.inotify.max_user_watches=1255360
RUN sudo sysctl -w fs.inotify.max_user_instances=2280

RUN echo "**** Creating user ****"
# [Optional] Delete existing user for ubuntu >= 23.04
# RUN userdel -r ubuntu
# [Optional] Set the default user. Omit if you want to keep the default as root.
RUN addgroup --gid 1000 $USERNAME
RUN adduser --disabled-password --gecos "" --uid $UID --gid $GID $USERNAME
RUN mkdir -p /home/$USERNAME
ENV HOME=/home/$USERNAME
RUN usermod -aG sudo $USERNAME
RUN echo '%sudo ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers
RUN mkdir -p /env

RUN chown -R $USERNAME /env
RUN chgrp -R $USERNAME /env
RUN chown -R $USERNAME $HOME
RUN chgrp -R $USERNAME $HOME

RUN echo "**** Installing Python ${PYTHON_VERSION} ****"

RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python3-pip \
    python${PYTHON_VERSION}-venv \
    python${PYTHON_VERSION}-dev

RUN echo "**** Continue as user ****"
USER $USERNAME

RUN echo "**** Installing Python packages (cached) ****"
RUN python${PYTHON_VERSION} -m venv /env
RUN source /env/bin/activate
ENV PATH="/env/bin:$PATH"
RUN echo which python

RUN pip install --upgrade pip
RUN pip install --no-cache-dir wheel
RUN pip install --no-cache-dir setuptools --upgrade --ignore-installed
RUN pip install --no-cache-dir --pre torch torchaudio torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
RUN pip install --no-cache-dir lightning tensorboard torch-tb-profiler opencv-python-headless nvidia-pyindex 

# ***************************************
# * Everything below will not be cached *
# ***************************************

RUN echo "**** Copying workspace ****"
COPY . /workspace
WORKDIR /workspace

ENV USERNAME=${USERNAME:-python-user}
RUN export USERNAME=$(whoami)

COPY ./.devcontainer/postCreateCommand.sh /docker-entrypoint.sh 
RUN sudo chmod +x /docker-entrypoint.sh

# RUN sudo chown -R $USERNAME:$USERNAME /workspace
# RUN sudo chown -R $USERNAME .
# RUN sudo chgrp -R $USERNAME .

RUN echo "**** Installing Python requirements ****"

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /workspace/requirements.txt

RUN echo "**** Entry point ****"
ENTRYPOINT ["/docker-entrypoint.sh"]

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************
