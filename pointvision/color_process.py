# -*- coding: utf-8 -*-
# @Time    : 2018/8/25 0025 20:56
# @Author  : Yinglin Zheng
# @Email   : zhengyinglin@stu.xmu.edu.cn
# @FileName: color_process.py
# @Software: PyCharm
# @Affiliation: XMU IPIC

def color_normalize(pc_color):
    '''
    :param pc_color: can be any shape as long as it only contain color
    :return:
    '''
    pc_color /= 127.5
    pc_color -= 1
    return pc_color


def color_unnormalize(pc_color):
    '''
    :param pc_color: can be any shape as long as it only contain color
    :return:
    '''
    pc_color += 1
    pc_color *= 127.5
    pc_color[pc_color > 255] = 255
    pc_color[pc_color < 0] = 0
    return pc_color
