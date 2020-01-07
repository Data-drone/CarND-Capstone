# Training a model for traffic light detection

This is built leveraging the Tensorflow Object Detection API and uses the Bosch Small Traffic Light dataset

The dataset was made compatible with the Tensorflow Object Detection API using the script in `dataset_tools/create_bosch_tf_record.py`

## Requirements:

TF 1.3.0 docker image:

docker pull tensorflow/tensorflow:1.3.0-devel-gpu-py3
docker run --gpus all -it -p 8888:8888 -v /home/brian/Workspace:/root/work --ipc=host -v /media/brian/extra_14:/root/work/external_data tensorflow/tensorflow:1.3.0-devel-gpu-py3

TF 1.4.0 docker image:

docker pull tensorflow/tensorflow:1.4.0-devel-gpu-py3
docker run --gpus all -it -p 8888:8888 -p 6006:6006 -v /home/brian/Workspace:/root/work --ipc=host -v /media/brian/extra_14:/root/work/external_data capstone_tl:tf_14


## Building Dataset

The builds out the tf records for the object detection api run from train_detector folder.

In order to make the models train properly, boxes which fall outside the 

```
python3 dataset_tools/create_bosch_tf_record.py \
    --data_dir=/root/work/external_data/bosch_traffic_light/ \
    --subset='train' \
    --output_path=/root/work/CarND-Capstone/ros/src/tl_detector/train_detector/data/bosch_train.record \
    --label_map_path=/root/work/CarND-Capstone/ros/src/tl_detector/train_detector/data/bosch_label_map.pbtxt

python3 dataset_tools/create_bosch_tf_record.py \
    --data_dir=/root/work/external_data/bosch_traffic_light/ \
    --subset='test' \
    --output_path=/root/work/CarND-Capstone/ros/src/tl_detector/train_detector/data/bosch_test.record \
    --label_map_path=/root/work/CarND-Capstone/ros/src/tl_detector/train_detector/data/bosch_label_map.pbtxt
```

## Training models with object detection API

running from the object detection api folder (`model/research`):

    export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

#### Model 1 
ssd_mobilenet_v1 trained from scratch

```
python3 object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=/root/work/CarND-Capstone/ros/src/tl_detector/train_detector/models/model_1/pipeline.config \
    --train_dir=/root/work/CarND-Capstone/ros/src/tl_detector/train_detector/models/model_1/train/ \
    --num_clones 2
```

#### Model 2
ssd_mobilenet_v1 extra training on top of v1

Export out for testing:

```
python3 object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path /root/work/CarND-Capstone/ros/src/tl_detector/train_detector/models/model_2/pipeline.config \
    --trained_checkpoint_prefix /root/work/CarND-Capstone/ros/src/tl_detector/train_detector/models/model_2/train/model.ckpt-150060 \
    --output_directory /root/work/CarND-Capstone/ros/src/tl_detector/train_detector/ssd_mobilenet_v1_export
```

#### Model 3
ssd_mobile_net_v1 with transfered weights from coco 2017/11/17 from model zoo

```
python3 object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path /root/work/CarND-Capstone/ros/src/tl_detector/train_detector/ssd_mobilenet_v1_coco_2017_11_17/pipeline.config \
    --trained_checkpoint_prefix /root/work/CarND-Capstone/ros/src/tl_detector/train_detector/ssd_mobilenet_v1_coco_2017_11_17/model.ckpt \
    --output_directory /root/work/CarND-Capstone/ros/src/tl_detector/train_detector/ssd_mobilenet_v1_pretrained_export

python3 object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=/root/work/CarND-Capstone/ros/src/tl_detector/train_detector/models/model_3/pipeline-model3.config \
    --train_dir=/root/work/CarND-Capstone/ros/src/tl_detector/train_detector/models/model_3/train/ \
    --num_clones 2

tensorboard --logdir=/root/work/CarND-Capstone/ros/src/tl_detector/train_detector/models/model_3/
```

#### Model 4
Faster RCNN model built with base weights from rcnn inception v2 trained on coco

```
python3 object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=/root/work/CarND-Capstone/ros/src/tl_detector/train_detector/models/model_4/pipeline-model4.config \
    --train_dir=/root/work/CarND-Capstone/ros/src/tl_detector/train_detector/models/model_4/train/ \
    --num_clones 2

python3 object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path /root/work/CarND-Capstone/ros/src/tl_detector/train_detector/models/model_4/pipeline-model4.config \
    --trained_checkpoint_prefix /root/work/CarND-Capstone/ros/src/tl_detector/train_detector/models/model_4/train/model.ckpt-9209 \
    --output_directory /root/work/CarND-Capstone/ros/src/tl_detector/train_detector/faster_rcnn_inception_v2_export

python3 object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=/root/work/CarND-Capstone/ros/src/tl_detector/train_detector/models/model_4/pipeline-model4.config \
    --train_dir=/root/work/CarND-Capstone/ros/src/tl_detector/train_detector/models/model_4/train/ \
    --num_clones 2
```

#### Model 5
Faster RCNN model built with base weights from rcnn resnet50 trained on coco

```
python3 object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=/root/work/CarND-Capstone/ros/src/tl_detector/train_detector/models/model_5/pipeline-model5.config \
    --train_dir=/root/work/CarND-Capstone/ros/src/tl_detector/train_detector/models/model_5/train/ \
    --num_clones 2
```