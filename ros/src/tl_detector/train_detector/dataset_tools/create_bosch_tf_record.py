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

import dataset_util
import label_map_util


flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'root directory to Bosch dataset.')
flags.DEFINE_string('subset', '', 'train or test')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', 'data/bosch_label_map.pbtxt',
                    'Path to label map proto')
FLAGS = flags.FLAGS

SETS = ['train', 'test']

bosch_label_dict = {'GreenStraightRight': 1, 'Red': 2, 'Yellow':3, 'RedStraightLeft':4, 
                    'RedLeft':5, 'GreenStraight':6, 'RedStraight': 7, 'RedRight': 8, 'off': 9,
                    'GreenStraightLeft': 10, 'GreenLeft': 11, 'GreenRight': 12, 'Green': 13}

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



def generate_data_dict(root_path):
    """parses folder and the returns the paths of all the images
    
    The files have been placed in different folders depending on which drive
    they came from so we need to find them to convert to record

    TODO Extension: sample across different drives for normalisation?

    Args:
      root_path: path to explore for subfolders

    Returns:
      Python Dictionary for doing look-ups
    
    """

    file_dict = {}
    for (dirpath, dirnames, filenames) in os.walk(root_path):
        for file in filenames:
            file_dict[file] = os.path.join(dirpath, file)

    return file_dict


def dict_to_tf_example(data,
                        dataset_dictionary,
                        label_map_dict, # do we need?
                        ignore_difficult_instances=False):
                        
    image_name = data['filename']
    image_path = dataset_dictionary[image_name]
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_png = fid.read()
    encoded_png_io = io.BytesIO(encoded_png)
    image = PIL.Image.open(encoded_png_io)
    key = hashlib.sha256(encoded_png).hexdigest()    

    width = int(data['size']['width'])
    height = int(data['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    
    for obj in data['object']:
        difficult = bool(int(obj['difficult']))
        if ignore_difficult_instances and difficult:
            continue

        difficult_obj.append(int(difficult))

        xmin_val = float(obj['bndbox']['xmin']) / width
        ymin_val = float(obj['bndbox']['ymin']) / height
        xmax_val = float(obj['bndbox']['xmax']) / width
        ymax_val = float(obj['bndbox']['ymax']) / height
        
        try:
            assert xmin_val < xmax_val 
            assert ymin_val < ymax_val
            assert xmin_val >= 0
            assert ymin_val >= 0
        except AssertionError as e:
            print(image_path)
            return None
        
        xmin.append(xmin_val)
        ymin.append(ymin_val)
        xmax.append(xmax_val)
        ymax.append(ymax_val)
        
        classes_text.append(obj['name'].encode('utf8'))
        assert bosch_label_dict[obj['name']] is not None
        classes.append(bosch_label_dict[obj['name']])
        #truncated.append(int(obj['truncated']))
        poses.append(obj['pose'].encode('utf8'))

        example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(
                data['filename'].encode('utf8')),
            'image/source_id': dataset_util.bytes_feature(
                data['filename'].encode('utf8')),
            'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
            'image/encoded': dataset_util.bytes_feature(encoded_png),
            'image/format': dataset_util.bytes_feature('png'.encode('utf8')),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes), 
            'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
            #'image/object/truncated': dataset_util.int64_list_feature(truncated),
            'image/object/view': dataset_util.bytes_list_feature(poses),
        }))


    return example

    
def main(_):
    if FLAGS.subset not in SETS:
        raise ValueError('set must be in : {}'.format(SETS))

    data_dir = os.path.join(FLAGS.data_dir, 'dataset_' + FLAGS.subset + '_rgb')

    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    logging.info('Reading bosch dataset')

    

    annotations_path = os.path.join(data_dir, 'annotations') # need check this

    annotations_list = os.listdir(annotations_path)

    data_path = os.path.join(data_dir, 'rgb', FLAGS.subset)
    data_dict = generate_data_dict(data_path)

    for idx, _file in enumerate(annotations_list):
        if idx % 100 == 0:
            logging.info('On image {0} of {1}'.format(idx, len(annotations_list)))

        path = os.path.join(annotations_path, _file)
        with tf.gfile.GFile(path, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)

        # we need to find the data... hmmmm.... 
        data = recursive_parse_xml_to_dict(xml)['annotaion']
        #print(data)

        if 'object' in data.keys(): 
            tf_example = dict_to_tf_example(data, data_dict, label_map_dict)

            if tf_example is not None:
                writer.write(tf_example.SerializeToString())

    writer.close()

if __name__ == '__main__':
    tf.app.run()