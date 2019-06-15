# -*- coding: utf-8 -*-

import torch
from torch import nn
from resnet import *
import pdb

def generate_model(opt):
    torch.manual_seed(opt.manual_seed)#乱数を固定
    model = resnet101().cuda()
    if opt.pretrain_path:
        print('loading pretrained model {}'.format(opt.pretrain_path))
        pretrain = torch.load(opt.pretrain_path)
        model.load_state_dict(pretrain['state_dict'])
        #model.load_state_dict(pretrain)
    model.fc_custom=nn.Linear(model.fc_custom.in_features, 10)
    model.fc_custom=model.fc_custom.cuda()
    return model

def test_generate_model(opt):
    torch.manual_seed(opt.manual_seed)#乱数を固定
    model = resnet101().cuda()
    if opt.pretrain_path:
        print('loading pretrained model {}'.format(opt.pretrain_path))
        pretrain = torch.load(opt.pretrain_path)
        model.load_state_dict(pretrain['state_dict'])
        #model.load_state_dict(pretrain)
    #model.fc_custom=nn.Linear(model.fc_custom.in_features, 10)
    #model.fc_custom=model.fc_custom.cuda()
    return model