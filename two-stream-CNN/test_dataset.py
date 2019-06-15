# -*- coding: utf-8 -*-

import torch
import torch.utils.data as data
from PIL import Image
import os
import functools
import json
import copy
import math
import pdb
from utils import load_value_file
import pdb

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def load_PReID_annotation_data(annotation_path):
    with open(annotation_path, 'r') as data_file:
        data = json.load(data_file)

    return data

def get_PReID_video_names_and_labels(data):
    video_names = []
    video_labels = []
    classnametoids = {}
    num = []
    day = []
    count = 0
    for video_name in data.keys():
        if data[video_name]["num"] in range(1,7):#numが1〜6だったら追加
            video_names.append(video_name)
            video_labels.append(data[video_name]["label"])
            num.append(data[video_name]["num"])
            day.append(data[video_name]["day"])
        if data[video_name]["num"] in range(11,17):#numが1〜6だったら追加
            video_names.append(video_name)
            video_labels.append(data[video_name]["label"])
            num.append(data[video_name]["num"])
            day.append(data[video_name]["day"])
        if data[video_name]["num"] in range(21,27):#numが1〜6だったら追加
            video_names.append(video_name)
            video_labels.append(data[video_name]["label"])
            num.append(data[video_name]["num"])
            day.append(data[video_name]["day"])
            if not data[video_name]["label"] in classnametoids:
                classnametoids[data[video_name]["label"]] = count
                count += 1

    return video_names,video_labels,num,day,classnametoids

def get_PReID_video_names_and_labels_test(data):
    video_names = []
    video_labels = []
    classnametoids = {}
    num = []
    day = []
    count = 0
    for video_name in data.keys():
        if data[video_name]["num"] in range(7,11):#numが7〜10だったら追加
            video_names.append(video_name)
            video_labels.append(data[video_name]["label"])
            num.append(data[video_name]["num"])
            day.append(data[video_name]["day"])
        if data[video_name]["num"] in range(17,21):#numが17〜20だったら追加
            video_names.append(video_name)
            video_labels.append(data[video_name]["label"])
            num.append(data[video_name]["num"])
            day.append(data[video_name]["day"])
        if data[video_name]["num"] in range(27,31):#numが27〜30だったら追加
            video_names.append(video_name)
            video_labels.append(data[video_name]["label"])
            num.append(data[video_name]["num"])
            day.append(data[video_name]["day"])
            if not data[video_name]["label"] in classnametoids:
                classnametoids[data[video_name]["label"]] = count
                count += 1

    return video_names,video_labels,num,day,classnametoids

def make_PReID_dataset(annotation_path, video_root):
    """
    アノテーションファイルの形式
    {
        動画番号:{
            "point":得点,
            "label":beginer or expert
            }
    }

    具体例
    {
        "32": {
            "point": 20,
            "label": "beginer"
        },
        "211": {
            "point": 90,
            "label": "expert"
        },
        .....
        .....
        "44": {
            "point": 40,
            "label": "beginer"
        }
    }
    """
    data = load_PReID_annotation_data(annotation_path)#annotationファイルの読み込み
    video_names,class_names,num,day,classnametoids = get_PReID_video_names_and_labels_test(data)#annotationから動画名の配列とクラス名の配列を作成，各要素番号は対応している
    class_idx = []
    for label in class_names:
        class_idx.append(classnametoids[label])#class_idxにラベルのidを追加していく
    dataset = []#このリストの各要素が１つの動画データになり，get_item内で参照される
    for i in range(len(class_idx)):
        #print('dataset loading [{}/{}]'.format(i, len(class_idx)))
        video_path = os.path.join(video_root, class_names[i], video_names[i])
        frame_num = len(os.listdir(video_path))
        if not os.path.exists(video_path):
            continue
        clip_num = 10
        sample = {
            'video': video_path,
            'label':class_names[i],
            'video_name': video_names[i],
            'num': num[i],
            'day': day[i],
            'label_idx':class_idx[i],
            "frame_indices":range(1,frame_num)
            }
        dataset.append(sample)
    return dataset

def get_default_PReID_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader

def PReID_video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, '{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video
    if len(video) < 16:
        for i in range(16-len(video)):
            video.append(image_loader(image_path))
    return video

def get_default_PReID_video_loader():
    image_loader = get_default_PReID_image_loader()
    return functools.partial(PReID_video_loader, image_loader=image_loader)

class PReID(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples

    """

    def __init__(self,
                 annotation_path,
                 video_root,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 get_loader=get_default_PReID_video_loader):
        self.data = make_PReID_dataset(annotation_path, video_root)#dataのlistを作る
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]['video']
        frame_indices = self.data[index]["frame_indices"]
        frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        clip = torch.squeeze(clip)
        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        target=torch.Tensor([target]).long()
        videoid = self.data[index]['video_name']
        #return clip, target, videoid
        return clip, target

    def __len__(self):
        return len(self.data)

def get_training_set(opt, spatial_transform, temporal_transform, target_transform):
    training_data_set = PReID(opt.annotation_path,
                           opt.video_root_path,
                           spatial_transform=spatial_transform,
                           temporal_transform=temporal_transform,
                           target_transform=target_transform)
    return training_data_set
