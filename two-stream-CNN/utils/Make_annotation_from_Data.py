# -*- coding: utf-8 -*-

import json
import pdb
import os
"""
動画から分類用アノテーションファイルを作成するコード
アノテーションのフォーマットは
{動画１：{"label":ラベル名}
 動画2:{"label":ラベル名}
           ・
           ・
           ・
 動画N{"label":ラベル名}
 }
"""


root_path = "./flowdata"

class_names = os.listdir(root_path)
annotation_dict = {}
for classess in class_names:
    videos = os.listdir(os.path.join(root_path, classess))
    for video in videos:
        annotation_dict[video] = {"label":classess}

with open("flow_annotation.json", 'w') as f:
json.dump(annotation_dict, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))