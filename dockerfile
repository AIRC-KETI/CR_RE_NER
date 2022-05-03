# docker build -t kr/service .

# docker tag kr/service 10.0.0.161:5000/service
# docker push 10.0.0.161:5000/service

# docker run --gpus all --rm -d -it -p 12345:5000 --name service kr/service
# or if you use nvidia-docker
# nvidia-docker run --rm -d -it -p 12345:5000 --name service kr/service

# docker -H 10.0.0.110:2375 run --gpus all --rm -d -it --network="dockernet" --ip="10.1.92.122" --name service 10.0.0.161:5000/service

# port: ketiair.com:10022 --> 10.1.92.122:5000

FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel
LABEL maintainer "KETI AIRC hyeontae <dchs504@keti.re.kr>"

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion locales rsync \
    libc6 libstdc++6 python-minimal tar curl net-tools apt-utils

RUN locale-gen en_US.UTF-8 && update-locale

ENV LC_ALL=en_US.UTF-8
ENV LANGUAGE=en_US:en

RUN pip install flask 
RUN pip install transformers
RUN pip install sentencepiece protobuf

ADD app /root/app

WORKDIR /root/app

# download move weights of model and move it to the ./app directory for fast build
# comment below script after above action
RUN cd /root/app && \
    python download_file_from_google_drive.py https://drive.google.com/file/d/1d9IGkiWZDCC9jK38U-bhgtScver_pcr8/view?usp=sharing output.tar.gz && \
    tar xvzf output.tar.gz

CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]

ENV SERVICE_PORT 5000
EXPOSE ${SERVICE_PORT}