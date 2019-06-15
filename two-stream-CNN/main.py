# -*- coding: utf-8 -*-

import os
import argparse
import sys
import json
import numpy as np
import torch
import pdb
from torch import nn
from torch import optim
from model import generate_model
from spatial_transforms import Compose, ToTensor
from target_transforms import ClassLabel
from temporal_transforms import TemporalRandomCrop4flow
from PIL import Image
from dataset import get_training_set
from train import train_epoch
from utils import Logger
from evaluation import test
from utils import AverageMeter

def opt():
    """
    学習率やモーメンタムは原さんの論文参照
    Kensho Hara, Hirokatsu Kataoka, and Yutaka Satoh,
    "Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?",
    Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 6546-6555, 2018.
    """
    parser = argparse.ArgumentParser(description="Classyfy beginer or expert in SoccerDB ")
    parser.add_argument("--n_classes", type=int, default=10, help="何クラス分類にするか")
    parser.add_argument("--resnet_shortcut", type=str, default="A", help="ResNetのshortcutをAにするかBにするか，ゼロパディング（A）にするか次元を増やす線形変換を学習するか（B）,34以下であればAを指定")
    parser.add_argument("--sample_size", type=int, default=224, help="フレームの縦横比")
    parser.add_argument("--sample_duration", type=int, default=16, help="時空間方向のフレーム数")
    parser.add_argument("--pretrain_path", type=str, default="data/weights/resnet101-ucfrgb.pth", help="時空間方向のフレーム数")
    #parser.add_argument("--pretrain_path", type=str, default="/home/pcd002/two_stream_cnn/result/learned_weight_day1_1-6/save_1000.pth", help="時空間方向のフレーム数")
    parser.add_argument("--model_depth", type=int, default="101", help="resnetの深さを何層にするか")
    parser.add_argument("--no_train", type=bool, default=False, help="trainを行わない場合（testなど）")
    parser.add_argument("--annotation_path", type=str, default="./data/annotations/day1-3_rgb.json", help="annotationファイルの相対パス")
    parser.add_argument("--video_root_path", type=str, default="./data/PReID_videos/Crop_rgb_224x224", help="annotationファイルの相対パス")
    parser.add_argument("--batch_size", type=int, default=64, help="バッチサイズ指定")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="学習率指定")
    parser.add_argument("--momentum", type=float, default=0.9, help="モーメンタム指定指定")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="モーメンタム指定")
    parser.add_argument("--n_epoch", type=int, default=1000, help="学習のepoch数を指定")
    parser.add_argument("--checkpoint", type=int, default=1000, help="何epochごとに重みを保存するか")
    parser.add_argument("--step_size", type=int, default=250, help="何epochごとに学習率を減らすか")
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = opt()#argsの読み出し
    args.arch = "ResNet-{}".format(args.model_depth)#実行するアーキテクチャを書き込む
    spatial_transform=Compose([
        ToTensor(),#1iterごとに読み込まれる各フレーム(PIL Image)をTensorへ変換する
    ])
    temporal_transform=TemporalRandomCrop4flow()#時間方向の前処理，今回はなし
    target_transform=ClassLabel()#学習する正解データ，２クラス分類なのでラベル
    accuracies=AverageMeter()#各回におけるaccとその平均
    with open(os.path.join("./result", 'opts.json'), 'w') as args_file:
        json.dump(vars(args), args_file)

    model = generate_model(args)#モデルの読み込み（pretrainがあれば重みも読み込んでおく）

    criterion = nn.CrossEntropyLoss()#損失関数の定義CrossEntropyLoss
    criterion.cuda()#criterionをcudaへ変更
    training_data=get_training_set(args, spatial_transform, temporal_transform, target_transform)#データローダに入力するデータセットの作成
    train_loader=torch.utils.data.DataLoader(
        training_data,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True
    )
    train_logger=Logger("./result/train.log", ["epoch", "loss", "acc", "lr"])
    optimizer=optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay)
    for epoch in range(1, args.n_epoch+1):
        train_epoch(epoch, train_loader, model, criterion, optimizer, args, train_logger)
    print("finish")
