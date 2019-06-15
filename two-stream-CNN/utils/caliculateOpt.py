# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import argparse
from multiprocessing import Pool
import pdb
"""
inputのRGBdatasetのフォルダ構成
data-classA-video1 _00001.jpg(RGB)
                  |_00002.jpg(RGB)
                  |_00003.jpg(RGB)
                         .
                         .
                         .
                  |_nnnnn.jpg(RGB)
outputのFLOWdatasetのフォルダ構成
flowdata-classA-video1 _00001.jpg(RGB画像の00001.jpgと00002.jpgのu方向のOpticalFlow画像)
                      |_00002.jpg(RGB画像の00001.jpgと00002.jpgのv方向のOpticalFlow画像)
                      |_00003.jpg(RGB画像の00002.jpgと00003.jpgのu方向のOpticalFlow画像)
                          .
                          .
                          .
                      |_nnnnn.jpg(RGB画像のnnnnn-1.jpgとnnnnn.jpgのv方向のOpticalFlow画像)
"""
def load_image(path):
    return cv2.imread(path)

def frame2flow(first_inputImage_path, second_inputImage_path):
    first_inputImage = cv2.cvtColor(load_image(first_inputImage_path),cv2.COLOR_BGR2GRAY)
    second_inputImage = cv2.cvtColor(load_image(second_inputImage_path),cv2.COLOR_BGR2GRAY)
    flow_Image = cv2.calcOpticalFlowFarneback(first_inputImage,second_inputImage, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    u_flowImage = flow_Image[:,:,0].astype("uint8")
    v_flowImage = flow_Image[:,:,1].astype("uint8")
    return u_flowImage, v_flowImage

def make_class_folder_paths(args):
    class_list = os.listdir(args.input_dataset_path)
    class_paths = []
    out_paths = []
    for className in class_list:
        class_paths.append(os.path.join(args.input_dataset_path,className))
        out_paths.append(os.path.join(args.output_dataset_path,className))
    return class_paths, out_paths

def make_output_folder(args):
    if not os.path.exists(args.output_dataset_path):
        os.mkdir(args.output_dataset_path)
    class_list = os.listdir(args.input_dataset_path)
    for className in class_list:
        output_class_path = os.path.join(args.output_dataset_path, className)
        input_class_path = os.path.join(args.input_dataset_path, className)
        if not os.path.exists(output_class_path):
            os.mkdir(output_class_path)
        for output_video in os.listdir(input_class_path):
            output_video_path = os.path.join(output_class_path, output_video.replace("cro","flow"))
            if not os.path.exists(output_video_path):
                os.mkdir(output_video_path)

def sortVideo(frames):
    sortedFrames = []
    for frame in frames:
        sortedFrames.append(int(frame.replace(".jpg","")))
    return sorted(sortedFrames)

def run_make_flowImage(class_path, output_classPath):
    input_videos = os.listdir(class_path)
    output_videos = os.listdir(output_classPath)
    for input_video, output_video in zip(input_videos, output_videos):
        input_video_path = os.path.join(class_path, input_video)
        output_video_path = os.path.join(output_classPath, output_video)
        print("Creating {} OpticalFlowImage".format(input_video))
        frames = os.listdir(input_video_path)
        frames = sortVideo(frames)
        for i, frame in enumerate(frames):
            framePath = os.path.join(input_video_path, "{:05d}.jpg".format(frame))
            if i == 0:
                count = 1
                prevFrame = framePath
            else:
                uflowImage, vflowImage = frame2flow(prevFrame, framePath)
                cv2.imwrite(os.path.join(output_video_path, "{:05d}.jpg".format(count)), uflowImage)
                count += 1
                cv2.imwrite(os.path.join(output_video_path, "{:05d}.jpg".format(count)), vflowImage)
                count += 1
                prevFrame = framePath

def opt():
    parser = argparse.ArgumentParser(description="rgbフレームをflowフレームに変更する")
    parser.add_argument("--input_dataset_path", type=str, default='/home/pcd002/Re-ID_movie/20190409/rgb_frame', help="入力rgbフレームデータセットのrootパス")
    parser.add_argument("--output_dataset_path", type=str, default='/home/pcd002/two_stream_cnn/data/PReID_videos/flow_data_1', help="出力flowUVフレームデータセットのrootパス")
    parser.add_argument("--num_worker", type=int, default=5, help="バッチサイズ指定")
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    args = opt()#argsの読み出し
    make_output_folder(args)
    input_class_folder_paths, output_class_folder_paths = make_class_folder_paths(args)

    for input_classPath ,output_classPath in zip(input_class_folder_paths, output_class_folder_paths):
        run_make_flowImage(input_classPath, output_classPath)
    """
    pool = Pool(args.num_worker)
    pool.map(run_make_flowImage, input_class_folder_paths, output_class_folder_paths)
    pool.close()
    """