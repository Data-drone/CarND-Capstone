# Light Detection for Carla

In order for Carla to be able to stop at traffic lights, we need to be able to detect traffic lights. In the simulator, traffic light signal status can be directly obtained from one of the subcribed topics.

On the real test track, we will leverage convolutional neural networks in order to detect traffic lights and the current state of the traffic lights. 

## Approach:

Given that we don't have a lot of data in order to train a full network from scratch, we will try to leverage other datasets and also utilise transfer learning. To make building a neural network easier, we will leverage Tensorflow's Object Detection API.

The [Bosch Traffic Light Dataset](https://hci.iwr.uni-heidelberg.de/node/6132) was leveraged in order to tune object detection models to focus in particular on traffic lights. Base models were pretrained models which has been trained on the coco dataset. 

We experimented with detection models built using [SSD models](https://arxiv.org/abs/1512.02325) and also [Faster-RCNN](https://arxiv.org/abs/1506.01497) models. SSD models run faster but did not perform as well as Faster-RCNN models for detecting small sized objects.

These models are then utilised in the `TLClassifier` class in `tl_classifier.py` in order to try and detect traffic lights from incoming images.

## Data Processing:

The Bosch dataset is provided as a series of png images from several different drives they are in `.png` format and accompanied by `.xml` annotations. In order to use these with the Tensorflow Object Detection API, it was necessary to convert these into tfrecord format. The script in `dataset_tools/create_bosch_tf_record.py` was written to do this.

The final records are stored in the `data` folder.

## Docker Images:

For training models, we used docker images as sometimes different pretrained models come in different tensorflow versions and we needed to use tf 1.3.0 in order to finally use the models with Carla.

### [TF 1.3.0 docker image](docker/Dockerfile_tf_1_3):

```Bash
# Base Tensorflow image
docker pull tensorflow/tensorflow:1.3.0-devel-gpu-py3

# Run image with nvidia docker 2.0
docker run --gpus all -it -p 8888:8888 -v /home/brian/Workspace:/root/work --ipc=host -v /media/brian/extra_14:/root/work/external_data tensorflow/tensorflow:1.3.0-devel-gpu-py3
```

### [TF 1.4.0 docker image](docker/Dockerfile):

```Bash
# Base Tensorflow image
docker pull tensorflow/tensorflow:1.4.0-devel-gpu-py3

# Run image with nvidia-docker 2.0
docker run --gpus all -it -p 8888:8888 -p 6006:6006 -v /home/brian/Workspace:/root/work --ipc=host -v /media/brian/extra_14:/root/work/external_data capstone_tl:tf_14
```

## Steps:

### Building Dataset

This builds out the tf records for the object detection api run from train_detector folder.
We found that the original authors of the Bosch Dataset also annotated and drew boxes for traffic lights that mostly fall outside the image. 

When running object detection, this can result in `NaN` loss so these were removed.


```Bash
# Create Train Dataset
python3 dataset_tools/create_bosch_tf_record.py \
    --data_dir=/root/work/external_data/bosch_traffic_light/ \
    --subset='train' \
    --output_path=/root/work/CarND-Capstone/ros/src/tl_detector/train_detector/data/bosch_train.record \
    --label_map_path=/root/work/CarND-Capstone/ros/src/tl_detector/train_detector/data/bosch_label_map.pbtxt

# Create Test Dataset
python3 dataset_tools/create_bosch_tf_record.py \
    --data_dir=/root/work/external_data/bosch_traffic_light/ \
    --subset='test' \
    --output_path=/root/work/CarND-Capstone/ros/src/tl_detector/train_detector/data/bosch_test.record \
    --label_map_path=/root/work/CarND-Capstone/ros/src/tl_detector/train_detector/data/bosch_label_map.pbtxt
```

### Training models with object detection API

In order to train models, the object detection api was utilised. The following codes need to be run from within object detection api folder (`model/research`):

``` Bash
# Required to enable the object detection API as per tf docs
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

#### Model 1 
ssd_mobilenet_v1 trained from scratch

```Bash
# Train Code
python3 object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=/root/work/CarND-Capstone/ros/src/tl_detector/train_detector/models/model_1/pipeline.config \
    --train_dir=/root/work/CarND-Capstone/ros/src/tl_detector/train_detector/models/model_1/train/ \
    --num_clones 2
```

#### Model 2
ssd_mobilenet_v1 extra training on top of v1.

```Bash
# train code
python3 object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=/root/work/CarND-Capstone/ros/src/tl_detector/train_detector/models/model_2/pipeline-model2.config \
    --trained_checkpoint_prefix /root/work/CarND-Capstone/ros/src/tl_detector/train_detector/models/model_1/train/model.ckpt-9209 \
    --train_dir=/root/work/CarND-Capstone/ros/src/tl_detector/train_detector/models/model_2/train/ \
    --num_clones 2

# exporting out model for use in TLClassifier
python3 object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path /root/work/CarND-Capstone/ros/src/tl_detector/train_detector/models/model_2/pipeline.config \
    --output_directory /root/work/CarND-Capstone/ros/src/tl_detector/train_detector/ssd_mobilenet_v1_export
```

#### Model 3
ssd_mobile_net_v1 with transfered weights from coco 2017/11/17 from model zoo

```Bash
# train code:
python3 object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=/root/work/CarND-Capstone/ros/src/tl_detector/train_detector/models/model_3/pipeline-model3.config \
    --train_dir=/root/work/CarND-Capstone/ros/src/tl_detector/train_detector/models/model_3/train/ \
    --num_clones 2

# export out model for use in TLClassifier
python3 object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path /root/work/CarND-Capstone/ros/src/tl_detector/train_detector/ssd_mobilenet_v1_coco_2017_11_17/pipeline.config \
    --trained_checkpoint_prefix /root/work/CarND-Capstone/ros/src/tl_detector/train_detector/ssd_mobilenet_v1_coco_2017_11_17/model.ckpt \
    --output_directory /root/work/CarND-Capstone/ros/src/tl_detector/train_detector/ssd_mobilenet_v1_pretrained_export

# track model training
tensorboard --logdir=/root/work/CarND-Capstone/ros/src/tl_detector/train_detector/models/model_3/
```

#### Model 4
Faster RCNN model built with base weights from rcnn inception v2 trained on coco

```Bash
# train model
python3 object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=/root/work/CarND-Capstone/ros/src/tl_detector/train_detector/models/model_4/pipeline-model4.config \
    --train_dir=/root/work/CarND-Capstone/ros/src/tl_detector/train_detector/models/model_4/train/ \
    --num_clones 2

# export out model for use in TLClassifier
python3 object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path /root/work/CarND-Capstone/ros/src/tl_detector/train_detector/models/model_4/pipeline-model4.config \
    --trained_checkpoint_prefix /root/work/CarND-Capstone/ros/src/tl_detector/train_detector/models/model_4/train/model.ckpt-9209 \
    --output_directory /root/work/CarND-Capstone/ros/src/tl_detector/train_detector/faster_rcnn_inception_v2_export
```

#### Model 5
Faster RCNN model built with base weights from rcnn resnet50 trained on coco

```Bash
# train model
python3 object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=/root/work/CarND-Capstone/ros/src/tl_detector/train_detector/models/model_5/pipeline-model5.config \
    --train_dir=/root/work/CarND-Capstone/ros/src/tl_detector/train_detector/models/model_5/train/ \
    --num_clones 2

# export out model for use in TLClassifier:
python3 object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path /root/work/CarND-Capstone/ros/src/tl_detector/train_detector/models/model_5/pipeline-model5.config \
    --trained_checkpoint_prefix /root/work/CarND-Capstone/ros/src/tl_detector/train_detector/models/model_5/train/model.ckpt-309 \
    --output_directory /root/work/CarND-Capstone/ros/src/tl_detector/train_detector/faster_rcnn_inception_v2_export

```

#### Model 6

Previous models on bosch set did not generalise out to the actual CarND Data from the testbag. 

```Bash
# train model
python3 object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=/root/work/CarND-Capstone/ros/src/tl_detector/train_detector/models/model_6/model_6.config \
    --train_dir=/root/work/CarND-Capstone/ros/src/tl_detector/train_detector/models/model_6/train/ \
    --num_clones 2
```