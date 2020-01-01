# Training a model for traffic light detection

## Requirements:

TF 1.3.0 docker image:

docker pull tensorflow/tensorflow:1.3.0-devel-gpu-py3

docker run --gpus all -it -p 8888:8888 -v /home/brian/Workspace:/root/work --ipc=host -v /media/brian/extra_14:/root/work/external_data tensorflow/tensorflow:1.3.0-devel-gpu-py3