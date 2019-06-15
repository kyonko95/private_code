import random
import math
import pdb


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, target):
        dst = []
        for t in self.transforms:
            dst.append(t(target))
            print(dst)
            pdb.set_trace()
        return dst


class ClassLabel(object):

    def __call__(self, target):
        return target['label_idx']


class VideoID(object):

    def __call__(self, target):
        return target['video_id']
