# Training a model for traffic light detection

## Requirements:

TF 1.3.0 docker image:

docker pull tensorflow/tensorflow:1.3.0-devel-gpu-py3

docker run --gpus all -it -p 8888:8888 -v /home/brian/Workspace:/root/work --ipc=host -v /media/brian/extra_14:/root/work/external_data tensorflow/tensorflow:1.3.0-devel-gpu-py3

#### New Docker Image
docker run --gpus all -it -p 8888:8888 -v /home/brian/Workspace:/root/work --ipc=host -v /media/brian/extra_14:/root/work/external_data capstone_tl:latest


#### Running train with object detection API

python3 object_detection/train.py --logtostderr --pipeline_config_path=/root/work/CarND-Capstone/ros/src/tl_detector/train_detector/models/model_1/pipeline.config --train_dir=/root/work/CarND-Capstone/ros/src/tl_detector/train_detector/models/model_1/train/ --num_clones 2

