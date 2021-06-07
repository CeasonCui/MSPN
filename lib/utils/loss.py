"""
@author: Wenbo Li
@contact: fenglinglwb@gmail.com
"""

import torch
import torch.nn as nn

class JointsL2Loss(nn.Module):
    def __init__(self, has_ohkm=False, topk=8, thresh1=1, thresh2=0):
        super(JointsL2Loss, self).__init__()
        self.has_ohkm = has_ohkm
        self.topk = topk
        self.t1 = thresh1
        self.t2 = thresh2
        method = 'none' if self.has_ohkm else 'mean'
        self.calculate = nn.MSELoss(reduction=method)

    def forward(self, output, valid, label):
        assert output.shape == label.shape
        batch_size = output.size(0)
        keypoint_num = output.size(1)
        loss = 0

        for i in range(batch_size):
            # print('output.shape',output[i].shape) #[17, 64, 48]
            pred = output[i].reshape(keypoint_num, -1) #[17,3072]
            # print('pred.shape', pred.shape)
            gt = label[i].reshape(keypoint_num, -1)
            if not self.has_ohkm: #前3层
                weight = torch.gt(valid[i], self.t1).float()
                gt = gt * weight #label矩阵里都是1，要乘上每个关键点的？

            tmp_loss = self.calculate(pred, gt)
            print('tmp_loss.shape', tmp_loss.shape)
            # print(tmp_loss)
            if self.has_ohkm: #最后1层
                tmp_loss = tmp_loss.mean(dim=1) 
                print('tmp_loss_mean.shape', tmp_loss.shape)
                weight = torch.gt(valid[i].squeeze(), self.t2).float()
                tmp_loss = tmp_loss * weight 
                topk_val, topk_id = torch.topk(tmp_loss, k=self.topk, dim=0,
                        sorted=False)
                print('topk_val.shape', topk_val.shape)
                sample_loss = topk_val.mean(dim=0)
                print('sample_loss', sample_loss)
            else:
                sample_loss = tmp_loss
            # print("sample_loss")
            # print(sample_loss)
            loss = loss + sample_loss 
            # print("loss")
            # print(loss)   
            # print("batch_size")
            # print(batch_size)         
            # print("loss_batch_size")
            # print(loss/batch_size)
        return loss / batch_size


if __name__ == '__main__':
    a = torch.ones(1, 17, 12, 12)
    b = torch.ones(1, 17, 12, 12)
    c = torch.ones(1, 17, 1) * 2
    loss = JointsL2Loss()
    # loss = JointsL2Loss(has_ohkm=True)
    device = torch.device('cuda')
    a = a.to(device)
    b = b.to(device)
    c = c.to(device)
    loss = loss.to(device)
    res = loss(a, c, b)
    print(res)


