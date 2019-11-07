import warnings
import argparse
import time
import torch
from tools import train, evaluate
from tools import cv_practical
from model.detector import circle_detector


def main(args):
    warnings.filterwarnings('ignore')
    # Set Primary Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Define Network
    net = circle_detector()
    print('Network Info:')
    print('Number of Network parameters: {0:.2f} Million'.format(sum(param.numel() for param in net.parameters()) / 1e6))
    net.to(device)
    print(net)
    
    if args.resume or args.phase == 'test':
        net.load_state_dict(torch.load(args.resumed_ckpt, map_location='cuda:0' if torch.cuda.is_available() else 'cpu')['net_state_dict'])
        print('Resumed Checkpoint: {} is Loaded!'.format(args.resumed_ckpt))

    if args.phase == 'train':
        train.train(args, net)
    else:
        net.eval()
        output = cv_practical.main.test_runner(args, net)
        print(output['message'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Scale Circle Detection Assignment')
    parser.add_argument('--phase', default='train', choices=['train', 'dev', 'test'], help='train/test mode selection',required=True, type=str)
    if parser.parse_known_args()[0].phase == 'train':
        parser.add_argument('--ckpt', default='./checkpoint/' + time.strftime("%Y-%m-%d-%H"), help='Path tp save the checkpoint', type=str)
        parser.add_argument('--train_batch_size', default=128, type=int)
        parser.add_argument('--train_set_size', default=1000, type=int)
        parser.add_argument('--dev_batch_size', default=128, type=int)
        parser.add_argument('--dev_set_size', default=1000, type=int)
        parser.add_argument('--lr', default=0.0001, help='Learning Rate', type=float)
        parser.add_argument('--weight_decay', default=0.0005, help='Optimizer Weight Decay', type=float)
        parser.add_argument('--num_workers', default=10, type=int)
        parser.add_argument('--start_epoch', default=0, type=int)
        parser.add_argument('--epochs', default=10, type=int)
        parser.add_argument('--loss_type', default='L2', choices=['L1', 'L2', 'Both'], help='The Loss Type to be used for training', type=str)
        if parser.parse_known_args()[0].loss_type == 'Both':
            parser.add_argument('--Lambda', default=0.1, help='Ratio of L1 to L2 loss', type=float)
        parser.add_argument('--evaluate_every_n_epoch', default=1, help='Testing Frequency of the Network', type=int)
        parser.add_argument('--mGPU', default=False, action='store_true', help='Multi-GPU Support')
    
    if parser.parse_known_args()[0].phase == 'test':
        parser.add_argument('--test_set_size', default=1000, type=int)
        parser.add_argument('--iou_threshold', default=0.7, type=float)
    
    parser.add_argument('--resume', default=False, action='store_true', help='Resume to specific checkpoint')
    
    if parser.parse_known_args()[0].phase == 'test' or parser.parse_known_args()[0].resume:
        parser.add_argument('--resumed_ckpt', default='', help='Path to the resumed checkpoint', type=str)
    
    args = parser.parse_args()
    main(args)


