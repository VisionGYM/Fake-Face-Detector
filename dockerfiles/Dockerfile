FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

WORKDIR /workspace/

ENV DEBIAN_FRONTEND nointeractive
ENV LC_ALL C.UTF-8
ENV TZ Asia/Tokyo
ENV PYTHONPATH $PYTHONPATH:/workspace/

RUN apt update \
    && apt upgrade -y \
    && apt install software-properties-common -y \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt install -y --no-install-recommends python3.10 python3-pip python-is-python3 git \
    && pip install --upgrade pip

RUN pip install -r requirements.txt