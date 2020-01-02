from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os
import glob

from lxml import etree
import PIL.Image
import tensorflow as tf

#from object_detection.utils import dataset_util
#from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'root directory to Bosch dataset.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
#flags.DEFINE_string('label_map_path', 'data/pascal_label_map.pbtxt',
#                    'Path to label map proto')
FLAGS = flags.FLAGS

def recursive_parse_xml_to_dict(xml):
    # from tf object detection api dataset_utils
  """Recursively parses XML contents to python dict.

  We assume that `object` tags are the only ones that can appear
  multiple times at the same level of a tree.

  Args:
    xml: xml tree obtained by parsing XML file contents using lxml.etree

  Returns:
    Python dictionary holding XML contents.
  """
  if not xml:
    return {xml.tag: xml.text}
  result = {}
  for child in xml:
    child_result = recursive_parse_xml_to_dict(child)
    if child.tag != 'object':
      result[child.tag] = child_result[child.tag]
    else:
      if child.tag not in result:
        result[child.tag] = []
      result[child.tag].append(child_result[child.tag])
  return {xml.tag: result}

def dict_to_tf_example(data,
                        dataset_directory,
                        label_map_dict,
                        )

                        pass

def main(_):

    data_dir = FLAGS.data_dir
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    #label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    logging.info('Reading bosch dataset')
    annotations_path = os.path.join(data_dir, 'annotations') # need check this

    annotations_list = os.listdir(annotations_path)

    for idx, _file in enumerate(annotations_list):
        if idx % 100 == 0:
            logging.info('On image {0} of {1}'.format(idx, len(_file)))

        #print('{0}'.format(_file))
        path = os.path.join(annotations_path, _file)
        with tf.gfile.GFile(path, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)

        # we need to find the data... hmmmm.... 
        data = recursive_parse_xml_to_dict(xml)['annotation']

        tf.example = dict

    writer.close()

if __name__ == '__main__':
    tf.app.run()