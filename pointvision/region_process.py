# -*- coding: utf-8 -*-
# @Time    : 2018/8/26 0026 8:50
# @Author  : Yinglin Zheng
# @Email   : zhengyinglin@stu.xmu.edu.cn
# @FileName: region_process.py
# @Software: PyCharm
# @Affiliation: XMU IPIC

import faiss
import random
from numba import jit
import numpy as np

@jit
def remove(dropedIdx,index_length):
    left_index=[]
    for i in range(index_length):
        if i not in dropedIdx:
            left_index.append(i)
    left_index = np.array(left_index)
    return left_index


def random_region_drop(pc_np, pc_c_np, ipc_num):
    '''
        :param pc_np : complete point cloud numpy, size=(n x Dim) Dim can be 3,6 or what ever, as long as the first three is x,y,z
        :param pc_c_np: complete point cloud color numpy, usually n x 3 ,aka, r,g,b
        :param ipc_num: number of point in the incomplete point cloud
        :return: incomplete point cloud and color and surface normal, size=Bx Dim x ipc_num
        '''

    dropout_num = pc_np.shape[0] - ipc_num
    if dropout_num == 0:
        return pc_np, pc_c_np

    center = random.randint(0, pc_np.shape[0] - 1)

    # make it into a gpu index

    # build index for faiss knn search
    coord = np.ascontiguousarray(pc_np[:, :3], dtype=np.float32)
    index = faiss.IndexFlatL2(3)  # the other index

    # here we specify METRIC_L2, by default it performs inner-product search
    index.add(coord)

    D, dropedIdx = index.search(np.array([coord[center]]), dropout_num)
    left_index=remove(dropedIdx,len(pc_np))

    ipc_np = np.take(pc_np, left_index, axis=0)
    ipc_c_np = np.take(pc_c_np, left_index, axis=0)

    return ipc_np, ipc_c_np

@jit
def jitter_point_cloud(data, sigma=0.02, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, jittered point clouds
    """
    N, C = data.shape
    assert (clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    jittered_data += data
    return jittered_data


def random_region_jitter_mask(pc_np, pc_color_np, ipc_num):
    '''
        :param pc_np : complete point cloud numpy, size=(n x Dim) Dim can be 3,6 or what ever, as long as the first three is x,y,z
        :param sn_np :surface normal usually n x 3 ,aka ,nx,ny,nz
        :param ipc_num: number of point in the incomplete point cloud
        :return: batch incomplete point cloud and color and surface normal, size=Bx Dim x ipc_num
        '''
    upc_np = pc_np.copy()
    upc_color_np = pc_color_np.copy()
    mask = np.ones((pc_np.shape[0], 1))
    dropout_num = pc_np.shape[0] - ipc_num
    if dropout_num == 0:
        return upc_np, upc_color_np, mask

    center = random.randint(0, upc_np.shape[0] - 1)

    # make it into a gpu index

    # build index for faiss knn search

    index = faiss.IndexFlatL2(3)  # the other index

    # here we specify METRIC_L2, by default it performs inner-product search
    index.add(upc_np)

    D, dropedIdx = index.search(np.array([upc_np[center]]), dropout_num)
    upc_np[dropedIdx] = jitter_point_cloud(pc_np[dropedIdx])
    upc_color_np[dropedIdx] = 0
    mask[dropedIdx] = 0

    return upc_np, upc_color_np, mask
