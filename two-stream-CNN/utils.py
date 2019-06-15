# -*- coding: utf-8 -*-

import csv


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)#tagetの0軸目のサイズから，batchサイズを取得
    _, pred = outputs.topk(1, 1, True)#モデルから出力されたバッチごとの各クラスの尤度ベクトルのうちもっとも尤度が高いクラスの要素番号を持つtensorを返す(バッチサイズ,要素番号).
    pred = pred.t()#(バッチサイズ,1)を(1,バッチサイズ)の各要素が推定されたクラスを表現するtensorを作成．例：tensor([[2, 1, 0]])
    correct = pred.eq(targets.view(1, -1))#targetsがtensor[2,0,1]のようなベクトルなのでpredと比較できるようにtensor[[2, 0, 1]]へ変換し，predとtargetを比較し正解を1不正解を0と表現したベクトルを返す．例:pred[[2, 0, 1]],target[[2,1,1]]であればcorrectは[[1, 0, 1]]
    n_correct_elems = correct.float().sum().item()#予測と教師が一致した数を計算

    return n_correct_elems / batch_size#バッチのうち予測と教師が一致した割合を返す
