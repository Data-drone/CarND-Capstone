import os
from styx_msgs.msg import TrafficLight
import tensorflow as tf

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        current_path = os.path.dirname(os.path.realpath(__file__))
        MODEL_DIR = 'faster_rcnn_inception_v2_export'
        self.model_file = os.path.join(current_path, MODEL_DIR, 'frozen_inference_graph.pb')

        self.model_graph = self.load_graph(self.model_file)


    def load_graph(self, graph_file):
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        
        #TODO implement light color prediction
        #with tf.Session(graph=self.model_graph) as sess:
        #    classification = sess.run([], feed_dict={image_tensor: image})

        # depending on the classification, we will return different TrafficLight variables
        return TrafficLight.UNKNOWN
