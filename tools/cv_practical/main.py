import numpy as np
from shapely.geometry.point import Point
from skimage.draw import circle_perimeter_aa
from model.detector import circle_detector
import torch
from tqdm import tqdm
import sys


def draw_circle(img, row, col, rad):
    rr, cc, val = circle_perimeter_aa(row, col, rad)
    valid = (
        (rr >= 0) &
        (rr < img.shape[0]) &
        (cc >= 0) &
        (cc < img.shape[1])
    )
    img[rr[valid], cc[valid]] = val[valid]


def noisy_circle(size, radius, noise):
    img = np.zeros((size, size), dtype=np.float)

    # Circle
    row = np.random.randint(size)
    col = np.random.randint(size)
    rad = np.random.randint(10, max(10, radius))
    draw_circle(img, row, col, rad)

    # Noise
    img += noise * np.random.rand(*img.shape)
    return (row, col, rad), img


def find_circle(net, img):
    # Put Image into torch tensor and configure dimensions
    img = torch.from_numpy(np.expand_dims(np.expand_dims(img, axis=0), axis=0))
    # Load to GPU for faster Computation
    if torch.cuda.is_available():
        img = img.cuda()
    with torch.no_grad():
        predicted = net(img.float())
    predicted = predicted.cpu().detach().numpy() * 200
    row, col, rad = predicted[0]
    return row, col, rad


def iou(params0, params1):
    row0, col0, rad0 = params0
    row1, col1, rad1 = params1

    shape0 = Point(row0, col0).buffer(rad0)
    shape1 = Point(row1, col1).buffer(rad1)

    return (
        shape0.intersection(shape1).area /
        shape0.union(shape1).area
    )


def test_runner(args, net):
    print('Used Arguments:')
    for key in args.__dict__.keys():        
        print(key + ':{}'.format(args.__dict__[key]))
    
    results = []
    with tqdm(total=args.test_set_size, ncols=0, file=sys.stdout, desc='Testing...') as pbar:
        for _ in range(args.test_set_size):
            params, img = noisy_circle(200, 50, 2)
            predicted = find_circle(net, img)
            results.append(iou(params, predicted))
            pbar.update()
    results = np.array(results)
    score = (results > args.iou_threshold).mean()
    output = {'message': 'Prediction score with IOU {0:.1f} on {1} test samples is {2}'.format(args.iou_threshold, args.test_set_size, score), 'accuracy': score}
    return output
