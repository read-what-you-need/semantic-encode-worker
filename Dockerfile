FROM nvidia/cuda:11.2.1-runtime-ubuntu20.04

LABEL maintainer="READ-NEED Core Maintainers <deeps@readneed.org>"

WORKDIR /code

RUN apt-get update && apt-get install -y \
        software-properties-common
    RUN add-apt-repository ppa:deadsnakes/ppa
    RUN apt-get update && apt-get install -y \
        python3.7 \
        python3-pip
    RUN python3.7 -m pip install pip
    RUN apt-get update && apt-get install -y \
        python3-distutils \
        python3-setuptools
    RUN python3.7 -m pip install pip --upgrade pip

COPY ./requirements.txt /code/requirements.txt

RUN pip3 install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./ /code/

CMD ["python3","-u", "main.py"]