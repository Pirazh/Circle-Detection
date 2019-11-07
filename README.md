# Circle Detection in Noisy Environment

The repository contains the code for circle parameter detection in the presence of noise. The code supports both training and testing the model.

## Getting Started 
Clone this repository with the following command:

```
git clone https://github.com/Pirazh/Circle-Detection
```

## requirements
The code is written in Python 3.6 with the [Pytorch](https://pytorch.org) deep learning framework. To install the dependancies run the following command:

```
pip install -r requirement.txt
```

## Test an already trained model
To test a trained model which is provided in `./checkpoint/2019-11-06-20` directory, you can run the bash script `run_test.sh` as follows:

```
sh run_test.sh
```

This file is configured to run the test code which loads the trained weights `./checkpoint/2019-11-06-20/best_checkpoint.pth.tar` into the circle detector and then access the script provided by Scale to evaluate the precision of the model on `1000` test samples. You can change the `test_set_size` in the `run_test.sh` to have a different size for the test set. The provided trained model yields the detection score of `0.93` with a model of around `700,000` parameters. 

Note that the code is designed in a way to support both GPU and CPU; however running the test on large test set size require a significant amount of time.

## Train a model
To train a model for detecting cicle's parameters in the presence of noise, you can should run the following:

```
sh run_train.sh
```

The current configuration is set to support Multi-GPUs for faster training. Also note that the training code initially generates train and development sets and you need around 32GB of ROM space in your local directory for this purpose. You can change configurations such as **Batch Size**, **Train/Dev Sets size**, **Number of Epochs** and **Single/Multi GPU support** in the `run_train.sh`. After training is done the trained model weights and `output.txt` containing all the logs can be accessed in `./checkpoint/FOLDER_NAMED_CURRENT_DATE_TIME`. 
