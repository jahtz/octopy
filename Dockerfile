FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.11

LABEL org.opencontainers.image.source=https://github.com/jahtz/octopy
LABEL org.opencontainers.image.description="Command line tool layout analysis and OCR of historical prints using Kraken."
LABEL org.opencontainers.image.licenses=APACHE-2.0

RUN apt-get update && apt-get install -y software-properties-common curl git && \
    add-apt-repository ppa:deadsnakes/ppa -y && apt-get update && \
    apt-get install -y build-essential python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python3-pip && \
    curl -O https://bootstrap.pypa.io/get-pip.py && python${PYTHON_VERSION} get-pip.py && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN git clone --single-branch --branch octopy https://github.com/jahtz/kraken && \
    pip${PYTHON_VERSION} install ./kraken

WORKDIR /build/octopy
COPY cli ./cli
COPY octopy ./octopy
COPY LICENSE .
COPY pyproject.toml .
COPY README.md .
RUN pip${PYTHON_VERSION} install .
RUN octopy --version

WORKDIR /data
ENTRYPOINT [ "octopy" ]
