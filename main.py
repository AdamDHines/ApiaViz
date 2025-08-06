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

def apianet_eval(args):
    # Initialize the evaluation class
    evaluator = EvalVision(args)
    evaluator.eval()

def apianet_train(args):
    # Initialize the training class
    trainer = TrainVision(args)
    trainer.train()

def parse_args():
    '''
    Define the base parameter parser (configurable by the user)
    '''
    parser = argparse.ArgumentParser(description="Args for default configuration")

    # Training or evaluation mode
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                        help='Mode to run: training or evaluation network')
    
    # Artificial or spiking mode
    parser.add_argument('--snn', action='store_true',
                        help='Use artificial neural network (default)')
    parser.add_argument('--num_steps', type=int, default=100,
                        help='Number of time steps for SNN simulation (default: 100)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train modules')
    parser.add_argument('--train_samples', type=int, default=100_000,
                        help='Number of training samples to use')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate for training')
    parser.add_argument('--models_dir', type=str, default='./apiaviz/models/',
                        help='Directory to save and load models')
    
    # Evaluation parameters
    parser.add_argument('--eval_samples', type=int, default=1,
                        help='Number of samples to use for evaluation')
    parser.add_argument('--eval_batch_size', type=int, default=7,
                        help='Batch size for evaluation')
    parser.add_argument('--outdir', type=str, default='./apiaviz/output/',
                        help='Directory to save evaluation results')
    parser.add_argument('--n_eval_plots', type=int, default=3,
                        help='Number of evaluation plots to generate')
    parser.add_argument('--eval_plot_random', action='store_true',
                        help='Use random sampling for evaluation plots')
    parser.add_argument('--ind_plots', type=str, default='batch',
                        choices=['ind', 'batch'], help='Plots over individual or batch samples')
    
    # Training dataset parameters
    parser.add_argument("--train_dataset", default="tiny", choices=["synthetic", "tiny"],
                        help="synthetic = VisionDataset,  tiny = Tiny-ImageNet (100k real images)")
    
    # Evaluation dataset parameters
    parser.add_argument("--eval_dataset", default="natural-scenes", choices=["synthetic", "faces", "natural-scenes"],
                        help="evaluation dataset to use")
    parser.add_argument('--green_pct_high', type=int, default=90,
                        help='Percentage of green in the dataset (for synthetic dataset)')
    parser.add_argument('--green_pct_low', type=int, default=10,
                        help='Percentage low of green in the dataset (for synthetic dataset)')
    
    # Model names
    parser.add_argument('--vision_model', type=str, default='VisionModel',
                        help='Name of the vision model for saving/loading')
    
    # Output base configuration
    args = parser.parse_args()
    args.snn = True
    # Run the user-chosen mode
    if args.mode == 'train':
        apianet_train(args)
    elif args.mode == 'eval':
        apianet_eval(args)

if __name__ == "__main__":
    args = parse_args()