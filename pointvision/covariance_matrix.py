# -*- coding: utf-8 -*-
# @Time    : 2018/8/24 0024 21:32
# @Author  : Yinglin Zheng
# @Email   : zhengyinglin@stu.xmu.edu.cn
# @FileName: covariance_matrix.py
# @Software: PyCharm
# @Affiliation: XMU IPIC

import torch
import faiss
import numpy as np
from numba import jit


def computeCovarianceMatrix(points):
    '''
    :param points: N*3
    :return:
    '''
    covarianceMatrix = torch.zeros(B, 9)
    means = points.mean(dim=0).unsqueeze(0).expand_as(points)
    for index in range(9):
        i = index // 3
        j = index % 3
        covarianceMatrix[index] = ((means[:, i] - points[:, i]) * (means[:, j] - points[:, j])).mean()
    return covarianceMatrix


# 建立NN索引
def build_nn_index(database):
    '''
    :param database: numpy array of Nx self.dimension
    :return: Faiss index, in CPU 在CPU当中的Faiss 索引
    '''
    index = faiss.IndexFlatL2(3)
    index.add(database)
    return index


# K-nearest neighbor graph
def build_min_idx(points, k):
    index = build_nn_index(points)
    _, I = index.search(points, k)  # I: numpy array of Nxk
    return I


@jit
def build_A_matrix_numba(min_idx):
    A = np.zeros((min_idx.shape[0], min_idx.shape[0]))  # 邻接矩阵
    for i, a in enumerate(A):
        a[min_idx[i]] = 1
    return torch.from_numpy(A).int()


def build_A_matrix_torch(min_idx):
    min_idx = torch.from_numpy(min_idx)  # I: numpy array of Nxk ，K近邻图

    min_idx_expand = min_idx.unsqueeze(1).expand(min_idx.shape[0], min_idx.shape[0],
                                                 min_idx.shape[1])  # NxNxk

    point_list = torch.arange(min_idx.shape[0]).unsqueeze(0).unsqueeze(2).expand_as(
        min_idx_expand).long()  # NxNxk

    A, _ = torch.eq(min_idx_expand, point_list).max(dim=2).int()  # NxN

    return A


# 计算每个点的协方差矩阵
def perPointCovarianceMatrix(points, k, input_channels=3):
    '''
    :param points: ndarray of N*input_channels
    :return: torch tensor of input_channels^2 *N
    :return min_idx k*N
    '''
    assert input_channels <= points.shape[1]
    points = np.ascontiguousarray(points[:, :input_channels])

    covarianceMatrix = torch.zeros(points.shape[0], input_channels ** 2)
    min_idx = torch.from_numpy(build_min_idx(points, k))  # I: numpy array of Nxk ，K近邻图

    # A=build_A_matrix_torch(min_idx)


    neighbors = torch.Tensor([np.take(points, i, axis=0) for i in min_idx])  # N*K*input_channels
    means = neighbors.mean(dim=1).unsqueeze(1).expand_as(neighbors)  # N*input_channels
    for index in range(input_channels ** 2):
        i = index // input_channels
        j = index % input_channels
        covarianceMatrix[:, index] = (
                (means[:, :, i] - neighbors[:, :, i]) * (means[:, :, j] - neighbors[:, :, j])).mean(dim=1)


    # input_channels^2 x N ,k x N
    return covarianceMatrix.transpose(0, 1), min_idx.transpose(0, 1)
