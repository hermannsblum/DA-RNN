# --------------------------------------------------------
# DA-RNN
# Copyright (c) 2016
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

from . import shapenet_scene
from . import shapenet_single
from . import gmu_scene
from . import rgbd_scene
from . import lov
import numpy as np

# shapenet dataset
for split in ['train', 'val']:
    name = 'shapenet_scene_{}'.format(split)
    print name
    __sets[name] = (lambda split=split:
            shapenet_scene(split))

for split in ['train', 'val']:
    name = 'shapenet_single_{}'.format(split)
    print name
    __sets[name] = (lambda split=split:
            shapenet_single(split))

# gmu scene dataset
for split in ['train', 'val']:
    name = 'gmu_scene_{}'.format(split)
    print name
    __sets[name] = (lambda split=split:
            gmu_scene(split))

# rgbd scene dataset
for split in ['train', 'val', 'trainval']:
    name = 'rgbd_scene_{}'.format(split)
    print name
    __sets[name] = (lambda split=split:
            rgbd_scene(split))

# lov dataset
for split in ['train', 'val']:
    name = 'lov_{}'.format(split)
    print name
    __sets[name] = (lambda split=split:
            lov(split))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
