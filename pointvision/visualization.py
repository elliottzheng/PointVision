# -*- coding: utf-8 -*-
# @Time    : 2018/8/19 0019 13:04
# @Author  : Yinglin Zheng
# @Email   : zhengyinglin@stu.xmu.edu.cn
# @FileName: visualization.py
# @Software: PyCharm
# @Affiliation: XMU IPIC

from pointvision.color_process import *
import torch
import numpy as np

def plot_color_pointcloud_with_som(pc, title, vis, win=None, som=None,unnormalize=False):
    '''
    :param pc: tensor of B x C x N in cuda, C>=6
    :param title:
    :param vis:
    :param win:
    :param som: tensor of B x C x N in cuda
    :param unnormalize:
    :return:
    '''
    pc = pc[0].cpu().numpy().transpose()
    if unnormalize:
        pc[:, 3:6] =color_unnormalize(pc_color=pc[:, 3:6])
    if som is not None:
        som_np= som[0].cpu().numpy().transpose()
        red = torch.Tensor([255, 0, 0]).unsqueeze(0).expand((som_np.shape[0],3)).numpy()
        som_np=np.hstack([som_np, red])
        pc=np.vstack([pc,som_np])

    win = vis.scatter(
        X=pc[:, :3],
        win=win,
        opts=dict(markercolor=pc[:, 3:6], markersize=2, title=title),
    )

    return win

def plot_pointcloud_with_som(pc, title, vis, win=None, som=None):
    pc = pc[0].detach().cpu().numpy().transpose()
    black = torch.Tensor([0, 0, 0]).unsqueeze(0).expand((pc.shape[0], 3)).numpy()
    pc=np.hstack([pc,black])
    if som is not None:
        som_np= som[0].detach().cpu().numpy().transpose()
        red = torch.Tensor([255, 0, 0]).unsqueeze(0).expand((som_np.shape[0],3)).numpy()
        som_np=np.hstack([som_np, red])
        pc=np.vstack([pc,som_np])

    win = vis.scatter(
        X=pc[:, :3],
        win=win,
        opts=dict(markercolor=pc[:, 3:6], markersize=2, title=title),
    )

    return win

def plot_loss(epoch,title,train_loss,test_loss,vis,win=None):
    y = train_loss.item()
    z = test_loss.item()
    win = vis.line(
        X=np.array([[epoch, epoch]]),
        Y=np.array([[y, z]]),
        win=win,
        opts=dict(legend=["train_loss", "test_loss"],
                  title=title),
        update=None if win is None else 'append'
    )

    return win
