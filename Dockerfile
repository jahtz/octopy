FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.11

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common libsm6 libxext6 libxrender-dev curl && \
    rm -rf /var/lib/apt/lists/*

# Install python
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y build-essential python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python3-pip && \
    curl -O https://bootstrap.pypa.io/get-pip.py && \
    python${PYTHON_VERSION} get-pip.py && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get install -y git

WORKDIR /build
RUN git clone --single-branch --branch octopy https://github.com/jahtz/kraken && \
    pip${PYTHON_VERSION} install ./kraken

WORKDIR /build/octopy
COPY src ./src
COPY LICENSE .
COPY pyproject.toml .
COPY README.md .
RUN pip${PYTHON_VERSION} install .

WORKDIR /workspace
ENTRYPOINT [ "octopy" ]
