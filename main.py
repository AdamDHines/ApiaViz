#MIT License

#Copyright (c) 2025 TBD

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

import argparse

from apiaviz.src.eval import EvalVision
from apiaviz.src.train import TrainVision
from apiaviz.src.logger import model_logger
from apiaviz.nav.gardenspoint import GardensPoint

def apianet_eval(args, logger, output_folder):
    if args.eval_dataset == "gardens-point":
        gp = GardensPoint()
        gp.run_vpr('day_left','day_right')
    else:
        # Initialize the evaluation class
        evaluator = EvalVision(args, logger, output_folder)
        evaluator.eval()

def apianet_train(args, logger, output_folder):
    # Initialize the training class
    trainer = TrainVision(args, logger, output_folder)
    trainer.train()

def parse_args():
    '''
    Define the base parameter parser (configurable by the user)
    '''
    parser = argparse.ArgumentParser(description="Args for default configuration")

    # Opertaion modes
    parser.add_argument('-m', '--mode', type=str, default='eval', choices=['train', 'eval'],
                        help='Mode to run: training or evaluation network')
    
    # Vision Module parameters
    parser.add_argument('-s', '--snn',  action='store_true',
                        help='Use artificial neural network (default)')
    parser.add_argument('--num_steps', type=int, default=25,
                        help='Number of time steps for SNN simulation (default: 100)')
    parser.add_argument('--patch_size', type=int, default=75,  
                        help='Size of the input patches')
    parser.add_argument('-vm', '--vision_model', type=str, default='VisionModel',
                        help='Name of the vision model for saving/loading')
    parser.add_argument('-svm', '--snn_vision_model', type=str, default='SNNVisionModel',
                        help='Name of the SNN vision model for saving/loading')
    
    # Training parameters
    parser.add_argument('-e', '--epochs', type=int, default=20,
                        help='Number of epochs to train modules')
    parser.add_argument('--train_samples', type=int, default=100_000,
                        help='Number of training samples to use')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for training')
    parser.add_argument('--training_dataset', type=str, default='tiny-imagenet',
                        help='Dataset to use for training')
    parser.add_argument('-bo', '--best_only', action='store_true',
                        help='Save only the best model during training')
    
    # Evaluation parameters
    parser.add_argument('--eval_samples', type=int, default=128,
                        help='Number of samples to use for evaluation')
    parser.add_argument('--eval_batch_size', type=int, default=128,
                        help='Batch size for evaluation')
    
    # Evaluation dataset parameters
    parser.add_argument('-d', '--eval_dataset', default="flowers", choices=["synthetic", "faces", "flowers", "variety", "17flowers", "gardens-point", "gardens-point-few"],
                        help="evaluation dataset to use")
    parser.add_argument('-sc', '--scanning', action='store_true',
                        help='Use scanning for the evaluation dataset')
    
    # Project directories
    parser.add_argument('--models_dir', type=str, default='./apiaviz/models/',
                        help='Directory to save and load models')
    parser.add_argument('--dataset_dir', type=str, default='./apiaviz/dataset/',
                        help='Directory where datasets are stored')
    parser.add_argument('--output_dir', type=str, default='./apiaviz/output/',
                        help='Directory to save evaluation results')
    
    # Output base configuration
    args = parser.parse_args()
    logger, output_dir = model_logger(args)
    
    if args.mode == 'train':
        apianet_train(args, logger, output_dir)
    elif args.mode == 'eval':
        apianet_eval(args, logger, output_dir)

if __name__ == "__main__":
    args = parse_args()