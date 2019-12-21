#
import torch.nn.functional as F

class AqpLoss(object):
    @staticmethod
    def nll(output, target):
        y_hat = F.softmax(output, dim=1)
        return -y_hat[range(target.shape[0]), target].mean()