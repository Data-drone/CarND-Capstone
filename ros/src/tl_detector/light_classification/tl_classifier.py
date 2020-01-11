import os
from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
from PIL import ImageDraw, ImageColor
import time

cmap = ImageColor.colormap
COLOR_LIST = sorted([c for c in cmap.keys()])

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        current_path = os.path.dirname(os.path.realpath(__file__))
        MODEL_DIR = 'faster_rcnn_inception_v2_export'
        self.model_file = os.path.join(current_path, MODEL_DIR, 'frozen_inference_graph.pb')

        self.model_graph = self.load_graph(self.model_file)

        self.image_tensor = self.model_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.model_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.model_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.model_graph.get_tensor_by_name('detection_classes:0')

        self.visualise = True


    def load_graph(self, graph_file):
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph

    def filter_boxes(self, min_score, boxes, scores, classes):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score:
                idxs.append(i)
        
        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes

    def draw_boxes(self, image, boxes, classes, thickness=4):
        """draw bounding boxes on the image"""
        draw = ImageDraw.Draw(image)
        for i in range(len(boxes)):
            bot, left, top, right = boxes[i, ...]
            class_id = int(classes[i])
            color = COLOR_LIST[class_id]
            draw.line([(left, top), (left, bot), (right, bot), (right, top), (left, top)], width=thickness, fill=color)

    def to_image_coords(self, boxes, height, width):
        """
        The original box coordinate output is normalized, i.e [0, 1].
        
        This converts it back to the original coordinate based on the image
        size.
        """
        box_coords = np.zeros_like(boxes)
        box_coords[:, 0] = boxes[:, 0] * height
        box_coords[:, 1] = boxes[:, 1] * width
        box_coords[:, 2] = boxes[:, 2] * height
        box_coords[:, 3] = boxes[:, 3] * width
        
        return box_coords

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)

        #TODO implement light color prediction
        with tf.Session(graph=self.model_graph) as sess:
            (boxes, scores, classes) = sess.run([self.detection_boxes, self.detection_scores, self.detection_classes], 
                                        feed_dict={self.image_tensor: image_np})

            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)

            confidence_cutoff = 0.1

            boxes, scores, classes = self.filter_boxes(confidence_cutoff, boxes, scores, classes)

            for record in classes:
                if record in [2,4,5,7,8]:

                    if self.visualise:
                        width, height = image.size
                        box_coords = self.to_image_coords(boxes, height, width)
                        self.draw_boxes(image, box_coords, classes)

                        # save image
                        name = "../../../../img_export/class_red-{}.png".format(time.time()*100)
                        image.save(name, "PNG")

                    return TrafficLight.RED

            

        # depending on the classification, we will return different TrafficLight variables
        return TrafficLight.UNKNOWN
