import torch

def make_optimizer(args, net):
    params = net.parameters()
    optimizer = torch.optim.Adam(
        params, lr= args.lr, weight_decay=args.weight_decay)

    return optimizer