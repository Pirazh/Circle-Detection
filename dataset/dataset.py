import torch
from torch.utils.data import Dataset
import os, sys
import numpy as np
from tqdm import tqdm
import tools.cv_practical.main as cv_tools


class CircleDataSet(Dataset):
    def __init__(self, phase='train', args=None):
        
        if not os.path.exists(os.path.join('./dataset', phase)):
            print('{} dataset does not exist, generating...'.format(phase))
            set_size = args.train_set_size if phase == 'train' else args.dev_set_size
            make_data(set_type=phase, set_size=set_size)

        if phase == 'train':
            annotation_file = './dataset/train/annotations.txt'
        else:
            annotation_file = './dataset/dev/annotations.txt'
        
        self.annotations = []
        with open(annotation_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_path = line.split(' ')[0]
                row, col, rad = line.split(' ')[1:]
                params = tuple([int(row), int(col), int(rad)])
                self.annotations.append([img_path, params])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, item):
        img_path, params = self.annotations[item]
        img = np.load(img_path)
        params = np.array(params)
        return torch.from_numpy(img).float(), torch.from_numpy(params).float() / 200


def make_data(set_type='train', set_size=1e4, size=200, max_radius=50, noise_level=2):

    # Check the directories to store the data
    if not os.path.exists(os.path.join('./dataset', set_type)):
        os.mkdir(os.path.join('./dataset', set_type))
        os.mkdir(os.path.join('./dataset', set_type, 'images'))
    else:
        print('The Dataset exists already!')
        return


    # Instantiate the annotation.txt file for target set
    with open('./dataset/{}/annotations.txt'.format(set_type), 'w+') as f:
        
        with tqdm(total=set_size, ncols=0, file=sys.stdout, desc='{} set generation'.format(set_type)) as pbar:
            
            for i in range(set_size):
                # Data Generation
                params, img = cv_tools.noisy_circle(size, max_radius, noise_level)
                img = np.expand_dims(img, axis=0)
                img_path = os.path.join('./dataset', set_type, 'images', '{:05}.npy'.format(i))
                np.save(img_path, img)
                f.write(img_path + ' ' + str(params[0]) + ' ' + str(params[1]) + ' ' + str(params[2]) + '\n')
                pbar.update()




        
