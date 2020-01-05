FROM tensorflow/tensorflow:1.4.0-devel-gpu-py3

LABEL maintainer="Brian Law <bpl.law@gmail.com>"

# add ffmpeg
RUN apt-get -y update && \
    apt-get -y upgrade

RUN apt-get install -y ffmpeg

# update jupyter
RUN pip install --upgrade jupyter && \
    pip install jupyterlab && \
    pip install pyyaml lxml

RUN apt-get -y install python3-tk

RUN pip install Pillow

ENV SHELL=bash

#ENTRYPOINT ["jupyter lab", "--allow-root"]