# testing container notes:
# docker run -v $PWD:/root/detector --rm -it tensorflow/tensorflow:1.3.0-devel
# docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone

import unittest
from tl_classifier import TLClassifier

def test_tl_classifier_init():
    new_classifier = TLClassifier()
    return new_classifier


if __name__ == '__main__':

    classifier = test_tl_classifier_init()