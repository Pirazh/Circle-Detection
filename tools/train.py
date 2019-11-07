import torch
from tools.solver import make_optimizer
from tools.utils import save_checkpoint, Chronometer, Logger
from tools import evaluate
from dataset.dataloader import make_dataloader
from layers.loss import Loss
from tqdm import tqdm
import os, sys


def train(args, net):
    # Get DataLoader
    data_loader = make_dataloader(args)
    
    # Get Optimizer
    optimizer = make_optimizer(args, net)
    
    # Get Criterion
    criterion = Loss(args=args)
    
    # Get Timer
    timer = Chronometer()
    
    # Get Logger
    logger = Logger(args=args)
    logger.print_net(net)

    # Check for Multi GPU Support
    if torch.cuda.device_count() > 1 and args.mGPU:
        net = torch.nn.DataParallel(net)
    
    # Create a directory for training files
    if not os.path.exists(args.ckpt):
        os.mkdir(args.ckpt)

    start_epoch = args.start_epoch
    if args.resume:
            checkpoint = torch.load(args.resumed_ckpt)
            start_epoch = checkpoint['epoch']

    best_accuracy = 0.0
    timer.set()
    for epoch in range(start_epoch, args.epochs):
        logger('Epoch: {}'.format(epoch + 1), prt=False)
        epoch_train_loss, is_best = 0.0, False
        
        with tqdm(total=len(data_loader), ncols=0, file=sys.stdout, desc='Epoch: {}'.format(epoch + 1)) as pbar:

            for i, in_batch in enumerate(data_loader):
                optimizer.zero_grad()
                in_data, target = in_batch
                # Load to GPU
                if torch.cuda.is_available():
                    in_data, target = in_data.cuda(), target.cuda()
                # Forward Pass
                predicted = net(in_data)
                # Backward Pass
                loss = criterion(predicted, target)
                epoch_train_loss += loss.item()
                loss.backward()
                optimizer.step()

                # Update Progressbar
                if i % 50 == 49:
                    logger('[Train loss/batch: {0:.4f}]'.format(loss.item()), prt=False)
                pbar.set_postfix(Loss=loss.item())
                pbar.update()

        epoch_train_loss /= len(data_loader)

        message = 'Average Training Loss : {0:.4f}'.format(epoch_train_loss)
        logger(message)

        # Check Performance of the trained Model on test set
        if epoch % args.evaluate_every_n_epoch == args.evaluate_every_n_epoch - 1:
            print('Network Evaluation...')
            net.eval()
            output = evaluate.evaluate(args, net)
            net.train()
            logger(output['message'])
            if output['accuracy'] > best_accuracy:
                best_accuracy = output['accuracy']
                is_best = True
            # save the checkpoint as best checkpoint so far
            save_checkpoint(
                {'epoch': epoch + 1,
                'net_state_dict': net.module.state_dict() if args.mGPU else net.state_dict()},
                is_best, filename=os.path.join(args.ckpt, 'checkpoint.pth.tar'),
                best_filename=os.path.join(args.ckpt, 'best_checkpoint.pth.tar'))
    
    timer.stop()
    message = 'Finished Trainig Session in {0} hours & {1} minutes, Best Accuracy Achieved: {2:.2f}\n'.format(int(timer.elapsed / 3600), int((timer.elapsed % 3600) / 60), best_accuracy)
    logger(message)
    logger.end()

