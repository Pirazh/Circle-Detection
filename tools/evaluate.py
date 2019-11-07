import torch
import sys
import numpy as np
from tools.cv_practical.main import iou
from dataset.dataloader import make_dataloader
from tqdm import tqdm


def evaluate(args, net):
    data_loader = make_dataloader(args, phase='evaluate')

    with tqdm(total=len(data_loader), ncols=0, file=sys.stdout, desc='Evaluation on Test Set...') as pbar:

        with torch.no_grad():
            for i, in_batch in enumerate(data_loader):
                in_data, target = in_batch

                if torch.cuda.is_available():
                    in_data = in_data.cuda()

                predicted = net(in_data)

                if i == 0:
                    target_np = target.numpy()
                    predicted_np = predicted.cpu().detach().numpy()
                else:
                    target_np = np.concatenate((target_np, target.numpy()), axis=0)
                    predicted_np = np.concatenate((predicted_np, predicted.cpu().detach().numpy()), axis=0)
                pbar.update()
        
        results = []
        for i in range(target_np.shape[0]):
            results.append(iou(tuple(target_np[i] * 200), tuple(predicted_np[i] * 200)))
        results = np.array(results)

        accuracy = (results > 0.7).mean()

    output = {'message': 'Dev Set Prediction Score with IOU 0.7 is {}'.format(accuracy), 'accuracy': accuracy}

    return output