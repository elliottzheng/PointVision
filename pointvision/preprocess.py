# -*- coding: utf-8 -*-
# @Time    : 2018/8/25 0025 20:56
# @Author  : Yinglin Zheng
# @Email   : zhengyinglin@stu.xmu.edu.cn
# @FileName: preprocess.py
# @Software: PyCharm
# @Affiliation: XMU IPIC

def color_normalize(pc_color_np):
    pc_color_np /= 127.5
    pc_color_np -= 1
    return pc_color_np


def color_unnormalize(pc_color_np):
    pc_color_np += 1
    pc_color_np *= 127.5
    pc_color_np[pc_color_np > 255] = 255
    pc_color_np[pc_color_np < 0] = 0
    return pc_color_np
