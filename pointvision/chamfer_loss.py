import torch
import torch.nn as nn
import numpy as np
import faiss


def robust_norm(var):
    '''
    :param var: Variable of BxCxHxW
    :return: p-norm of BxCxW
    '''
    result = ((var ** 2).sum(dim=2) + 1e-8).sqrt()
    # result = (var ** 2).sum(dim=2)

    # try to make the points less dense, caused by the backward loss
    # result = result.clamp(min=7e-3, max=None)
    return result


class ChamferLoss3D(nn.Module):
    def __init__(self, opt):
        super(ChamferLoss3D, self).__init__()
        self.opt = opt
        self.dimension = 3  # you can modified this to change dimension

        self.k = 1  # which means finding the nearest point

        # we need only a StandardGpuResources per GPU
        self.res = faiss.StandardGpuResources()
        self.res.setTempMemoryFraction(0.1)
        self.flat_config = faiss.GpuIndexFlatConfig()
        self.flat_config.device = opt.gpu_id

    def build_nn_index(self, database):
        '''
        :param database: numpy array of Nx3
        :return: Faiss index, in CPU
        '''
        # index = faiss.GpuIndexFlatL2(self.res, self.dimension, self.flat_config)  # dimension is 3
        index_cpu = faiss.IndexFlatL2(self.dimension)
        index = faiss.index_cpu_to_gpu(self.res, self.opt.gpu_id, index_cpu)
        index.add(database)
        return index

    def search_nn(self, index, query, k):
        '''
        :param index: Faiss index
        :param query: numpy array of Nx3
        :return: D: Variable of Nxk, type FloatTensor, in GPU
                 I: Variable of Nxk, type LongTensor, in GPU
        '''
        D, I = index.search(query, k)

        D_tensor = torch.from_numpy(np.ascontiguousarray(D))
        I_tensor = torch.from_numpy(np.ascontiguousarray(I).astype(np.int64))

        if self.opt.gpu_id >= 0:
            D_tensor = D_tensor.to(self.opt.device)
            I_tensor = I_tensor.to(self.opt.device)

        return D_tensor, I_tensor

    # this is the traditional forward, you can take it as a layer
    def forward(self, predict_pc, gt_pc):

        '''
        :param predict_pc: Bxinput_channelsxM Variable in GPU
        :param gt_pc: Bxinput_channelsxN Variable in GPU, ground truth
        :return:
        '''
        predict_pc = predict_pc[:, :3, :].ascontiguousarray()
        gt_pc = gt_pc[:, :3, :].ascontiguousarray()

        predict_pc_size = predict_pc.size()
        gt_pc_size = gt_pc.size()

        predict_pc_np = np.ascontiguousarray(
            torch.transpose(predict_pc.data.clone(), 1, 2).cpu().numpy())  # BxMxinput_channels
        gt_pc_np = np.ascontiguousarray(
            torch.transpose(gt_pc.data.clone(), 1, 2).cpu().numpy())  # Bx N x input_channels

        # selected_gt: B x k x input_channels xM
        selected_gt_by_predict = torch.FloatTensor(predict_pc_size[0], self.k, predict_pc_size[1], predict_pc_size[2])

        # selected_predict: B x k x input_channels xN ,these are some place holders
        selected_predict_by_gt = torch.FloatTensor(gt_pc_size[0], self.k, gt_pc_size[1], gt_pc_size[2])

        if self.opt.gpu_id >= 0:
            selected_gt_by_predict = selected_gt_by_predict.to(self.opt.device)
            selected_predict_by_gt = selected_predict_by_gt.to(self.opt.device)

        # process each point cloud independently.
        for i in range(predict_pc_np.shape[0]):
            index_predict = self.build_nn_index(predict_pc_np[i])
            index_gt = self.build_nn_index(gt_pc_np[i])

            # database is gt_pc, predict_pc -> gt_pc -----------------------------------------------------------
            _, I_tensor = self.search_nn(index_gt, predict_pc_np[i], self.k)

            # process nearest k neighbors
            for k in range(self.k):
                selected_gt_by_predict[i, k, ...] = gt_pc[i].index_select(1, I_tensor[:, k])

            # database is predict_pc, gt_pc -> predict_pc -------------------------------------------------------
            _, I_tensor = self.search_nn(index_predict, gt_pc_np[i], self.k)

            # process nearest k neighbors
            for k in range(self.k):
                selected_predict_by_gt[i, k, ...] = predict_pc[i].index_select(1, I_tensor[:, k])

        # compute loss ===================================================
        # selected_gt(Bxkxinput_channelsxM) vs predict_pc(Bxinput_channelsxM)
        forward_loss_element = robust_norm(
            selected_gt_by_predict - predict_pc.unsqueeze(1).expand_as(selected_gt_by_predict))
        forward_loss = forward_loss_element.mean()  # this is what i said, we need to take mean

        backward_loss_element = robust_norm(
            selected_predict_by_gt - gt_pc.unsqueeze(1).expand_as(selected_predict_by_gt))  # BxkxN
        backward_loss = backward_loss_element.mean()

        return forward_loss + backward_loss

    def __call__(self, predict_pc, gt_pc):

        loss = self.forward(predict_pc, gt_pc)

        return loss


class ChamferLoss6D(nn.Module):
    def __init__(self, opt):
        super(ChamferLoss6D, self).__init__()
        self.opt = opt
        self.dimension = 6  # you can modified this to change dimension

        self.k = 1  # which means finding the nearest point

        # we need only a StandardGpuResources per GPU
        self.res = faiss.StandardGpuResources()
        self.res.setTempMemoryFraction(0.1)
        self.flat_config = faiss.GpuIndexFlatConfig()
        self.flat_config.device = opt.gpu_id

    def build_nn_index(self, database):
        '''
        :param database: numpy array of Nx3
        :return: Faiss index, in CPU
        '''
        # index = faiss.GpuIndexFlatL2(self.res, self.dimension, self.flat_config)  # dimension is 3
        index_cpu = faiss.IndexFlatL2(self.dimension)
        index = faiss.index_cpu_to_gpu(self.res, self.opt.gpu_id, index_cpu)
        index.add(database)
        return index

    def search_nn(self, index, query, k):
        '''
        :param index: Faiss index
        :param query: numpy array of Nx3
        :return: D: Variable of Nxk, type FloatTensor, in GPU
                 I: Variable of Nxk, type LongTensor, in GPU
        '''
        D, I = index.search(query, k)

        D_tensor = torch.from_numpy(np.ascontiguousarray(D))
        I_tensor = torch.from_numpy(np.ascontiguousarray(I).astype(np.int64))

        if self.opt.gpu_id >= 0:
            D_tensor = D_tensor.to(self.opt.device)
            I_tensor = I_tensor.to(self.opt.device)

        return D_tensor, I_tensor

    # this is the traditional forward, you can take it as a layer
    def forward(self, predict_pc, gt_pc):

        '''
        :param predict_pc: Bxinput_channelsxM Variable in GPU
        :param gt_pc: Bxinput_channelsxN Variable in GPU, ground truth
        :return:
        '''

        predict_pc_size = predict_pc.size()
        gt_pc_size = gt_pc.size()

        predict_pc_np = np.ascontiguousarray(
            torch.transpose(predict_pc.data.clone(), 1, 2).cpu().numpy())  # BxMxinput_channels
        gt_pc_np = np.ascontiguousarray(
            torch.transpose(gt_pc.data.clone(), 1, 2).cpu().numpy())  # Bx N x input_channels

        # selected_gt: B x k x input_channels xM
        selected_gt_by_predict = torch.FloatTensor(predict_pc_size[0], self.k, predict_pc_size[1], predict_pc_size[2])

        # selected_predict: B x k x input_channels xN ,these are some place holders
        selected_predict_by_gt = torch.FloatTensor(gt_pc_size[0], self.k, gt_pc_size[1], gt_pc_size[2])

        if self.opt.gpu_id >= 0:
            selected_gt_by_predict = selected_gt_by_predict.to(self.opt.device)
            selected_predict_by_gt = selected_predict_by_gt.to(self.opt.device)

        # process each point cloud independently.
        for i in range(predict_pc_np.shape[0]):
            index_predict = self.build_nn_index(predict_pc_np[i])
            index_gt = self.build_nn_index(gt_pc_np[i])

            # database is gt_pc, predict_pc -> gt_pc -----------------------------------------------------------
            _, I_tensor = self.search_nn(index_gt, predict_pc_np[i], self.k)

            # process nearest k neighbors
            for k in range(self.k):
                selected_gt_by_predict[i, k, ...] = gt_pc[i].index_select(1, I_tensor[:, k])

            # database is predict_pc, gt_pc -> predict_pc -------------------------------------------------------
            _, I_tensor = self.search_nn(index_predict, gt_pc_np[i], self.k)

            # process nearest k neighbors
            for k in range(self.k):
                selected_predict_by_gt[i, k, ...] = predict_pc[i].index_select(1, I_tensor[:, k])

        # compute loss ===================================================
        # selected_gt(Bxkxinput_channelsxM) vs predict_pc(Bxinput_channelsxM)
        forward_loss_element = robust_norm(
            selected_gt_by_predict - predict_pc.unsqueeze(1).expand_as(selected_gt_by_predict))
        forward_loss = forward_loss_element.mean()  # this is what i said, we need to take mean
        # forward_loss_array = forward_loss_element.mean(dim=1).mean(dim=1)

        # selected_predict(Bxkxinput_channelsxN) vs gt_pc(Bxinput_channelsxN)
        backward_loss_element = robust_norm(
            selected_predict_by_gt - gt_pc.unsqueeze(1).expand_as(selected_predict_by_gt))  # BxkxN
        backward_loss = backward_loss_element.mean()
        # backward_loss_array = backward_loss_element.mean(dim=1).mean(dim=1)
        # loss_array = forward_loss_array + backward_loss_array
        return forward_loss + backward_loss  # + self.sparsity_loss

    def __call__(self, predict_pc, gt_pc):
        # start_time = time.time()
        loss = self.forward(predict_pc, gt_pc)
        # print(time.time()-start_time)
        return loss
