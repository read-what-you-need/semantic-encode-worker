ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:22.02-py3
FROM ${FROM_IMAGE_NAME}

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN pip install --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex.git#egg=apex

## FROM python:3.9

LABEL maintainer="READ-NEED Core Maintainers <deeps@readneed.org>"

WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./ /code/

CMD ["python","-u", "main.py"]