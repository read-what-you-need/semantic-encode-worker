FROM nvidia/cuda:11.2.1-runtime-ubuntu20.04

LABEL maintainer="READ-NEED Core Maintainers <deeps@readneed.org>"

WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./ /code/

CMD ["python","-u", "main.py"]