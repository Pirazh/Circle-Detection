from torch.utils.data import DataLoader
from .dataset import CircleDataSet


def make_dataloader(args, phase='train'):
    if phase == 'evaluate':
        phase = 'dev' 
    target_set = CircleDataSet(args=args, phase=phase)
    shuffle = True if args.phase == 'train' else False
    batch_size = args.train_batch_size if args.phase == 'train' else args.dev_batch_size
    data_loader = DataLoader(target_set,
                                batch_size=batch_size, 
                                shuffle=shuffle, 
                                num_workers=args.num_workers)
    return data_loader
