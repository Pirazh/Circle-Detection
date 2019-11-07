import torch.nn as nn


class Loss(object):
    
    def __init__(self, args):
        self.is_single = True
        if args.loss_type == 'L1':
            self.loss = nn.L1Loss()
        elif args.loss_type == 'L2':
            self.loss = nn.MSELoss()
        else:
            self.l1 = nn.L1Loss()
            self.l2 = nn.MSELoss()
            self.ratio = args.Lambda
            self.is_single = False

    def __call__(self, predicted, target):
        if self.is_single:
            loss = self.loss(predicted, target)
        else:
            loss = self.ratio * self.l1(predicted, target) + self.l2(predicted, target) 
        return loss