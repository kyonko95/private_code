# -*- coding: utf-8 -*-

import torch
import time
import os
import sys
import pdb
from utils import AverageMeter, calculate_accuracy

def train_epoch(epoch, data_loader, model, criterion, optimizer, opt, epoch_logger):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time=AverageMeter()#このepochにおける１iterにかかる時間
    losses=AverageMeter()#このepochにおける1iterのlossとlossの平均を保持
    accuracies=AverageMeter()#このepochにおける1iterのaccuraciesとaccuraciesの平均を保持

    device=torch.device("cuda")#マルチGPUを使う場合の呪文
    end_time = time.time()
    #以下学習フロー
    for i, (inputs, targets, video_ids) in enumerate(data_loader):
        model.zero_grad()#モデルの勾配をゼロに
        inputs = inputs.to(device)#マルチGPUの場合の手続き，これをしないと各GPUに分けてデータをのせることができない
        targets = targets.squeeze(1)#tensor(10,1)からtensor(10)の１次元テンソルへ変換
        targets = targets.cuda()#cudaへ
        outputs = model(inputs)#inputs：（バッチサイズ， チャンネル数（３）, フレーム数(16), width(114), height(114)）をモデルに入力し，outputs：(バッチサイズ, 各クラス尤度)を返すforward計算
        loss = criterion(outputs, targets)#outputsとtargetsのLOSSを計算，クラスエントロピーなので正解次元の推定尤度のlogにマイナスを乗じた値のバッチごとの平均がLOSSになる
        acc = calculate_accuracy(outputs, targets)#outputsとtargetsからバッチの中でどれだけ正解したか出力

        losses.update(loss.item(), inputs.size(0))#このiterにおけるlossとこのepochにおけるこれまでのiterにおけるlossの平均を計算し保持
        accuracies.update(acc, inputs.size(0))#このiterにおけるaccとこのepochにおけるこれまでのiterにおけるaccの平均を計算し保持

        optimizer.zero_grad()#optimizerの勾配をゼロに
        loss.backward()#backward計算で勾配を算出
        optimizer.step()#算出した勾配から重みを更新

        batch_time.update(time.time() - end_time)#このiterでかかった時間を算出しこのepochにおけるこれまでのiterでかかった時間の平均を計算し保持
        end_time = time.time()#時間を更新

        #各種情報を表示
        """
        print('Epoch: [{0}][{1}/{2}]\t'.format(
                  epoch,
                  i + 1,
                  len(data_loader)
                  ))
        """
        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  loss=losses,
                  acc=accuracies))

    #epoch数，このepochにおけるlossの平均，このepochにおけるacc平均，このepochにおける学習率をtrain.logへ書き込む
    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'acc': accuracies.avg,
        'lr': optimizer.param_groups[0]['lr']
    })


    if epoch % opt.checkpoint == 0:
        save_file_path = os.path.join("./result/learned_weight_day1-3_1-6_rgb",
                                      'save_{}.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
            'arch': opt.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)
