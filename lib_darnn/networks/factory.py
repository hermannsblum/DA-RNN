# --------------------------------------------------------
# DA-RNN
# Copyright (c) 2016 CVGL Stanford
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

from . import vgg16
from . import vgg16_convs
from . import resnet50
import tensorflow as tf
from ..fcn.config import cfg

if cfg.TRAIN.SINGLE_FRAME:
    if cfg.NETWORK == 'VGG16':
        __sets['vgg16_convs'] = vgg16_convs(cfg.INPUT, cfg.TRAIN.NUM_CLASSES, cfg.TRAIN.NUM_UNITS, cfg.TRAIN.SCALES_BASE, cfg.TRAIN.VERTEX_REG, cfg.TRAIN.TRAINABLE)
    if cfg.NETWORK == 'RESNET50':
        __sets['resnet50'] = resnet50(cfg.INPUT, cfg.TRAIN.NUM_CLASSES, cfg.TRAIN.SCALES_BASE)
    if cfg.NETWORK == 'FCN8VGG':
        __sets['fcn8_vgg'] = fcn8_vgg(cfg.TRAIN.NUM_CLASSES, cfg.TRAIN.MODEL_PATH)
else:
    __sets['vgg16'] = vgg16(cfg.INPUT, cfg.TRAIN.NUM_STEPS, cfg.TRAIN.NUM_CLASSES, cfg.TRAIN.NUM_UNITS, cfg.TRAIN.SCALES_BASE)

def get_network(name):
    """Get a network by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown network: {}'.format(name))
    return __sets[name]

def list_networks():
    """List all registered imdbs."""
    return __sets.keys()
