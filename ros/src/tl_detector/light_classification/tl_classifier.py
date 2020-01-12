import os
from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
from PIL import Image, ImageColor, ImageDraw
import time
import cv2
import rospy


cmap = ImageColor.colormap
COLOR_LIST = sorted([c for c in cmap.keys()])

class TLClassifier(object):
    def __init__(self, model_dir, is_site):
        #TODO load classifier
        current_path = os.path.dirname(os.path.realpath(__file__))
        #MODEL_DIR = 'faster_rcnn_inception_v2_export'
        self.model_file = os.path.join(current_path, model_dir, 'frozen_inference_graph.pb')

        self.model_graph = self.load_graph(self.model_file)

        self.image_tensor = self.model_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.model_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.model_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.model_graph.get_tensor_by_name('detection_classes:0')

        self.visualise = False

        # we need different logic for different model for now
        self.is_site = is_site 
        
    def detect_site(self, record):
        if record in [2,4,5,7,8]:
            
            rospy.logwarn('red detected - site')
            return True

        else:

            return False


    def detect_coco(self, record, image):
        if record == 10:

            avg_color_per_row = np.average(image, axis=0)
            avg_color = np.average(avg_color_per_row, axis=0)

            #rospy.logwarn('color: {}'.format(avg_color))

            if (avg_color[0] > avg_color[1]) and (avg_color[0] > avg_color[2]):
                rospy.logwarn('red detected - coco')
                return True
        else:

            return False


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
        draw = Image.fromarray(image.astype('uint8'), 'RGB')
        draw = ImageDraw.Draw(draw)
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
        np_sing_image = np.asarray(image, dtype=np.uint8)
        image_np = np.expand_dims(np_sing_image, 0)
        height, width, channels = image.shape
        #rospy.logwarn('H:{0} W:{1} C:{2}'.format(height, width, channels))

        #TODO implement light color prediction
        with tf.Session(graph=self.model_graph) as sess:
            (boxes, scores, classes) = sess.run([self.detection_boxes, self.detection_scores, self.detection_classes], 
                                        feed_dict={self.image_tensor: image_np})

            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)

            confidence_cutoff = 0.2

            boxes, scores, classes = self.filter_boxes(confidence_cutoff, boxes, scores, classes)

            if self.visualise and len(classes) > 0:
                
                box_coords = self.to_image_coords(boxes, height, width)
                self.draw_boxes(np_sing_image, box_coords, classes)

                # save image
                rospy.logwarn('saving images')
                name = "../../../../img_export/classifier-{0}.png".format(time.time()*100)
                cv2.imwrite(name, np_sing_image)
                #image.save(name, "PNG")

            count = 0

            if len(classes) > 1:

                box_coords = self.to_image_coords(boxes, height, width)

                for index, record in enumerate(classes):

                    box_co = box_coords[index]
                    #scores = scores[index]

                    # make this a function?
                    if self.is_site:

                        result = self.detect_site(record)
                    else:
                        #rospy.logwarn('box coords: {}'.format(box_co))
                        #rospy.logwarn('image np shape: {}'.format(image_np.shape))
                        #rospy.logwarn('image type: {}'.format(image_np.type))
                        boxed_image = np_sing_image[int(box_co[0]):int(box_co[2]), 
                                                int(box_co[1]):int(box_co[3])]
                        #rospy.logwarn('box shape: {}'.format(boxed_image.shape))
                        result = self.detect_coco(record, boxed_image)

                    if result:
                        count += 1

                if count > 1:    
                    return TrafficLight.RED

            

        # depending on the classification, we will return different TrafficLight variables
        return TrafficLight.UNKNOWN
