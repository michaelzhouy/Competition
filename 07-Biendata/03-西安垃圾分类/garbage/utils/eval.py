from __future__ import print_function, absolute_import

__all__ = ['accuracy', 'precision']
import torch

# def accuracy(output, target, topk=(1,)):
#     """Computes the precision@k for the specified values of k"""
#     maxk = max(topk)
#     batch_size = target.size(0)
#
#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))
#
#     res = []
#     for k in topk:
#         correct_k = correct[:k].view(1, -1).float().sum(0, keepdim=True)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # topk返回一个元组, 元组的第一个元素为最大值, 第一个元素为最大值所在的索引
        # 这里应该是top3
        _, pred = output.topk(maxk, dim=1)
        pred = pred.t()  # 转置
        # expand_as()维度扩展为pred的维度
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        # print(correct.shape)
        res = []
        for k in topk:
            # print(correct[:k].shape)
            correct_k = correct[:k].float().sum()
            res.append(correct_k.mul_(100.0 / batch_size))

        # print(res)
        return res

def precision(output, target):
    pass