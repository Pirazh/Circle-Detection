import shutil
import torch
import numpy as np
import time


class Chronometer():
    def __init__(self):
        self.elapsed = 0
        self.start = 0
        self.end = 0

    def set(self):
        self.start = time.time()

    def stop(self):
        self.end = time.time()
        self.elapsed = (self.end - self.start)

    def reset(self):
        self.start, self.end, self.elapsed = 0, 0, 0


class Logger():
    def __init__(self, args):
        self.args = args
        self.messages = []
        self.save_path = args.ckpt + '/output.txt'
        self.messages.append('Training Session on ' + time.strftime("%Y%m%d-%H") + '\n')
        self.messages.append('Used Arguments:\n')
        print('Used Arguments:')
        for key in args.__dict__.keys():
                self.messages.append(key + ':{}\n'.format(args.__dict__[key]))
                print(key + ':{}'.format(args.__dict__[key]))
        
    def __call__(self, message, prt=True, skip=True):
        self.messages.append(message)
        if skip:
            self.messages.append('\n')
        if prt:
            print(message)

    def end(self):
        with open(self.save_path, 'w+') as f:
            for msg in self.messages:
                f.write(msg)
            f.close()

    def print_net(self, net):
        self.messages.append('\n')
        self.messages.append('Network Info: \n')
        self.messages.append('Number of Network parameters: {0:.2f} Million\n'.format(sum(param.numel() for param in net.parameters()) / 1e6))
        net_text = str(net)
        self.messages.append(net_text)
        self.messages.append('\n')


def save_checkpoint(state, is_best, filename='./checkpoint/checkpoint.pth.tar',
                    best_filename='./checkpoint/model_best.pth.tar'):

    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)
