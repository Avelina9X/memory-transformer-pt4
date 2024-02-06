# FROM nvcr.io/nvidia/tensorflow:23.03-tf2-py3
# FROM nvcr.io/nvidia/tensorflow:23.04-tf2-py3
FROM nvcr.io/nvidia/pytorch:24.01-py3
 
ARG local_uid
ARG local_user

RUN adduser --uid ${local_uid} --gecos "" --disabled-password ${local_user}

WORKDIR /home/${local_user}
WORKDIR /home/${local_user}/memory-transformer-pt4
WORKDIR /data/${local_user}/.hfcache
WORKDIR /data/${local_user}/the_pile

RUN chmod +rwx /data/lhk3/.hfcache
RUN chmod +rwx /data/lhk3/the_pile

# USER ${local_user}

ENV PATH="/home/${local_user}/.local/bin:${PATH}"

RUN apt install ninja-build

# COPY requirements.txt ./

RUN pip install --upgrade pip

RUN pip install wcmatch

RUN pip install wandb
RUN pip install zstandard
RUN pip install rich

RUN pip install transformers
RUN pip install datasets
RUN pip install diffusers

RUN pip install torcheval
RUN pip install einops
RUN pip install flash-attn --no-build-isolation --upgrade