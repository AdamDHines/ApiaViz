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
    parser.add_argument('-m', '--mode', type=str, default='train', choices=['train', 'eval'],
                        help='Mode to run: training or evaluation network')
    parser.add_argument('--train_stage', type=str, default='backbone', choices=['backbone', 'lobula_plate', 'projection'],
                        help='Training stage to run when mode=train')
    
    # Vision Module parameters
    parser.add_argument('-s', '--snn',  action='store_true',
                        help='Use artificial neural network (default)')
    parser.add_argument('--deterministic', action='store_true',
                        help='Enable fully deterministic training at the cost of speed')
    parser.add_argument('--num_steps', type=int, default=25,
                        help='Number of time steps for SNN simulation (default: 100)')
    parser.add_argument('--patch_size', type=int, default=75,  
                        help='Size of the input patches')
    parser.add_argument('-vm', '--vision_model', type=str, default='VisionModel',
                        help='Name of the vision model for saving/loading')
    parser.add_argument('-svm', '--snn_vision_model', type=str, default='SNNVisionModel',
                        help='Name of the SNN vision model for saving/loading')
    parser.add_argument('--lobula_plate_model', type=str, default='VisionModel_LobulaPlate',
                        help='Name of the lobula plate fine-tuned checkpoint to save')
    parser.add_argument('--projection_model', type=str, default='VisionProjection',
                        help='Name of the VPN and Kenyon projection checkpoint to save')
    
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
                        help='Dataset to use for training, e.g. tiny-imagenet or wildscenes2d')
    parser.add_argument('-bo', '--best_only', action='store_true',
                        help='Save only the best model during training')
    parser.add_argument('--backbone_checkpoint', type=str, default='',
                        help='Optional pretrained checkpoint path or stem for lobula plate fine-tuning')
    parser.add_argument('--spatial_supervision', type=str, default='synthetic_shift',
                        choices=['synthetic_shift'],
                        help='Spatial supervision mode for lobula plate fine-tuning')
    parser.add_argument('--spatial_image_size', type=int, default=64,
                        help='Input size for dense lobula plate fine-tuning')
    parser.add_argument('--spatial_max_shift', type=int, default=8,
                        help='Maximum translation in pixels for dense lobula plate fine-tuning')
    parser.add_argument('--min_spatial_shift', type=int, default=0,
                        help='Minimum translation magnitude in pixels for dense lobula plate fine-tuning')
    parser.add_argument('--spatial_crop_padding', type=int, default=0,
                        help='Deprecated padding argument retained for compatibility')
    parser.add_argument('--dense_temperature', type=float, default=0.1,
                        help='Temperature used for dense spatial contrastive loss')
    parser.add_argument('--dense_samples', type=int, default=128,
                        help='Sampled dense correspondence locations per image')
    parser.add_argument('--dense_loss_weight', type=float, default=1.0,
                        help='Weight for the dense spatial contrastive loss')
    parser.add_argument('--place_loss_weight', type=float, default=1.0,
                        help='Weight for cross-traverse place-matching contrastive loss in pose-pair mode')
    parser.add_argument('--pose_loss_weight', type=float, default=1.0,
                        help='Weight for relative pose regression loss in pose-pair mode')
    parser.add_argument('--shift_loss_weight', type=float, default=0.5,
                        help='Weight for the centroid-based shift supervision loss')
    parser.add_argument('--shift_loss_mid_weight', type=float, default=0.5,
                        help='Mid-stage weight for centroid-based shift supervision')
    parser.add_argument('--shift_loss_final_weight', type=float, default=0.5,
                        help='Late-stage weight for centroid-based shift supervision')
    parser.add_argument('--shift_loss_mid_epoch', type=int, default=0,
                        help='Epoch after which shift loss anneals to the mid-stage weight')
    parser.add_argument('--shift_loss_final_epoch', type=int, default=0,
                        help='Epoch after which shift loss anneals to the final-stage weight')
    parser.add_argument('--shift_curriculum_warmup_epochs', type=int, default=0,
                        help='Number of warmup epochs using the smallest translation range')
    parser.add_argument('--shift_curriculum_mid_epochs', type=int, default=0,
                        help='Number of curriculum epochs before switching to the full translation range')
    parser.add_argument('--shift_curriculum_warmup_max', type=int, default=3,
                        help='Maximum translation during the curriculum warmup stage')
    parser.add_argument('--shift_curriculum_mid_max', type=int, default=6,
                        help='Maximum translation during the curriculum mid stage')
    parser.add_argument('--val_split', type=float, default=0.0,
                        help='Fraction of spatial training images to reserve for validation')
    parser.add_argument('--val_samples', type=int, default=1024,
                        help='Number of validation samples to draw per epoch for spatial fine-tuning')
    parser.add_argument('--spatial_val_batch_size', type=int, default=0,
                        help='Batch size for spatial validation; 0 uses the training batch size')
    parser.add_argument('--early_stop_patience', type=int, default=4,
                        help='Number of epochs without validation dense-top1 improvement before stopping')
    parser.add_argument('--early_stop_min_delta', type=float, default=0.001,
                        help='Minimum validation dense-top1 improvement needed to reset early stopping')
    parser.add_argument('--unfreeze_lobula_epoch', type=int, default=0,
                        help='Epoch at which to unfreeze the top lobula layers during spatial fine-tuning; 0 disables it')
    parser.add_argument('--lobula_lr_scale', type=float, default=0.1,
                        help='Learning-rate multiplier for the partially unfrozen top lobula layers')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data-loader workers')
    parser.add_argument('--split_seed', type=int, default=1337,
                        help='Seed used for train/validation splits in spatial fine-tuning')
    parser.add_argument('--spatial_train_split_file', type=str, default='',
                        help='Optional file listing spatial-training frames relative to the dataset root')
    parser.add_argument('--spatial_val_split_file', type=str, default='',
                        help='Optional file listing spatial-validation frames relative to the dataset root')
    parser.add_argument('--wildscenes_match_radius_m', type=float, default=3.0,
                        help='Maximum XY distance in meters for cross-traverse WildScenes positive pairs')
    parser.add_argument('--wildscenes_yaw_threshold_deg', type=float, default=20.0,
                        help='Maximum yaw difference in degrees for cross-traverse WildScenes positive pairs')
    parser.add_argument('--wildscenes_max_candidates', type=int, default=4,
                        help='Maximum matched positives to retain per WildScenes anchor frame')
    parser.add_argument('--wildscenes_pose_pool_size', type=int, default=4,
                        help='Adaptive pooling size used by the pose-relation head on lobula plate maps')
    parser.add_argument('--wildscenes_cache_dir', type=str, default='./apiaviz/dataset/cache/wildscenes2d',
                        help='Directory for cached pre-resized WildScenes RGB frames')
    parser.add_argument('--wildscenes_cache_overwrite', action='store_true',
                        help='Rebuild the cached pre-resized WildScenes frames even if they already exist')
    parser.add_argument('--wildscenes_disable_resized_cache', action='store_true',
                        help='Disable the pre-resized WildScenes image cache and decode original images on the fly')
    parser.add_argument('--projection_near_max_shift', type=int, default=2,
                        help='Maximum shift in pixels for near-positive projection tuples')
    parser.add_argument('--projection_near_min_shift', type=int, default=0,
                        help='Minimum shift in pixels for near-positive projection tuples')
    parser.add_argument('--projection_far_max_shift', type=int, default=8,
                        help='Maximum shift in pixels for far-positive projection tuples')
    parser.add_argument('--projection_far_min_shift', type=int, default=4,
                        help='Minimum shift in pixels for far-positive projection tuples')
    parser.add_argument('--projection_vpn_dim', type=int, default=128,
                        help='Descriptor dimension for each VPN branch')
    parser.add_argument('--projection_spatial_pool_size', type=int, default=4,
                        help='Pooling size used by the spatial VPN branch')
    parser.add_argument('--projection_spatial_token_dim', type=int, default=64,
                        help='Token dimension used inside the spatial and conjunctive VPN branches')
    parser.add_argument('--projection_kc_dim', type=int, default=2048,
                        help='Number of Kenyon cells in the projection stage')
    parser.add_argument('--projection_kc_fan_in', type=int, default=8,
                        help='Sparse fan-in for the Kenyon projection layer')
    parser.add_argument('--projection_kc_sparsity', type=float, default=0.03,
                        help='Target sparsity for the Kenyon k-WTA stage')
    parser.add_argument('--projection_feature_loss_weight', type=float, default=1.0,
                        help='Weight for the invariant feature VPN objective')
    parser.add_argument('--projection_shift_loss_weight', type=float, default=1.0,
                        help='Weight for the spatial shift-regression objective')
    parser.add_argument('--projection_kc_loss_weight', type=float, default=1.0,
                        help='Weight for the Kenyon ordering objective')
    parser.add_argument('--projection_balance_loss_weight', type=float, default=0.05,
                        help='Weight for the Kenyon load-balancing regularizer')
    parser.add_argument('--projection_pose_margin', type=float, default=0.10,
                        help='Required similarity gap between near and far Kenyon codes')
    parser.add_argument('--projection_negative_margin', type=float, default=0.05,
                        help='Required similarity gap between far and negative Kenyon codes')
    parser.add_argument('--projection_preview_samples', type=int, default=24,
                        help='Number of deterministic validation tuples used for representation plots')
    
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
