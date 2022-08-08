# docker build -t kr/service .

# docker tag kr/service 10.0.0.161:5000/service
# docker push 10.0.0.161:5000/service

# docker run --gpus all --rm -d -it -p 12345:5000 --name service kr/service
# or if you use nvidia-docker
# nvidia-docker run --rm -d -it -p 12345:5000 --name service kr/service

# docker -H 10.0.0.110:2375 run --gpus all --rm -d -it --network="dockernet" --ip="10.1.92.122" --name service 10.0.0.161:5000/service

# port: ketiair.com:10022 --> 10.1.92.122:5000

FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
LABEL maintainer "KETI AIRC hyeontae <dchs504@gmail.com>"

ENV DEBIAN_FRONTEND noninteractive

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-get update && apt-get install -y --no-install-recommends wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion locales rsync \
    libc6 libstdc++6 python-minimal tar curl net-tools apt-utils
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
RUN mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN locale-gen en_US.UTF-8 && update-locale

ENV LC_ALL=en_US.UTF-8
ENV LANGUAGE=en_US:en

RUN pip install flask 
RUN pip install transformers
RUN pip install sentencepiece protobuf

COPY . /root/NLP_modules

WORKDIR /root/NLP_modules

CMD ["python", "app/app.py"]

ENV SERVICE_PORT 5000
EXPOSE ${SERVICE_PORT}