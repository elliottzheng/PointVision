# -*- coding: utf-8 -*-
# @Time    : 2018/8/25 0025 17:04
# @Author  : Yinglin Zheng
# @Email   : zhengyinglin@stu.xmu.edu.cn
# @FileName: network_io.py
# @Software: PyCharm
# @Affiliation: XMU IPIC
import torch
import os

def save_network(dir,network, network_label, epoch_label, device):
    save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
    save_path = os.path.join(dir, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    network.to(device)
