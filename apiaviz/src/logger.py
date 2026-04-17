import os, logging, logging, torch, sys

from datetime import datetime

def model_logger(args):
    """
    Configure the model logger
    """   
    now = datetime.now()
    output_base_folder = args.output_dir
    # Determine subfolders based on model and dataset
    if args.snn:
        subfolder_model = 'snn'
    else:
        subfolder_model = 'ann'
    if args.mode == 'train':
        subfolder_mode = 'train'
        output_folder = os.path.join(output_base_folder, subfolder_mode, subfolder_model, now.strftime("%d%m%y-%H-%M-%S"))
    else:
        subfolder_mode = 'eval'
        subfolder_dataset = args.eval_dataset
        output_folder = os.path.join(output_base_folder, subfolder_mode, subfolder_model, subfolder_dataset, now.strftime("%d%m%y-%H-%M-%S"))
    
    # Create the specific output folder
    os.makedirs(output_folder, exist_ok=True)

    # 1. Get your custom logger
    logger = logging.getLogger("apiaviz")

    # 2. STOP messages from propagating to the root logger
    #    This is the most important line.
    logger.propagate = False

    # 3. Set the lowest-possible level for the logger.
    #    This allows it to pass all messages to its handlers.
    logger.setLevel(logging.DEBUG)

    # 4. If handlers already exist, clear them to avoid duplication
    #    This is important if this code is ever re-run in the same process.
    if logger.hasHandlers():
        logger.handlers.clear()

    # 5. Create the CONSOLE handler and set its level to INFO
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # 6. Create the formatter for the console output (message only)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)

    # 7. Create the FILE handler and set its level to DEBUG
    #    (Replace 'apiaviz.log' with your desired file path if needed)
    file_handler = logging.FileHandler(f'{output_folder}/apiaviz.log', mode='w') # 'w' to overwrite the log on each run
    file_handler.setLevel(logging.DEBUG)

    # 8. Create the formatter for the file output (with timestamp, level, etc.)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)-8s - %(message)s")
    file_handler.setFormatter(file_formatter)

    # 9. Add the handlers to YOUR logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    def handle_exception(exc_type, exc_value, exc_traceback):
        """
        Log any uncaught exceptions.
        This is a global exception handler.
        """
        if issubclass(exc_type, KeyboardInterrupt):
            # Don't log KeyboardInterrupt, let the program exit
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    # 11. Assign the handler to sys.excepthook
    sys.excepthook = handle_exception
    
    # Add the logger to the console (if specified)
    logger.info('')
    logger.info('   █████████              ███            █████   █████  ███')            
    logger.info('  ███░░░░░███            ░░░            ░░███   ░░███  ░░░')
    logger.info(' ░███    ░███  ████████  ████   ██████   ░███    ░███  ████   █████████')
    logger.info(" ░███████████ ░░███░░███░░███  ░░░░░███  ░███    ░███ ░░███  ░█░░░░███            .' '.            __")
    logger.info(" ░███░░░░░███  ░███ ░███ ░███   ███████  ░░███   ███   ░███  ░   ███░    .        .   .           (__\_")  
    logger.info(" ░███    ░███  ░███ ░███ ░███  ███░░███   ░░░█████░    ░███    ███░   █   .         .         . -{{_(|8)")
    logger.info(" █████   █████ ░███████  █████░░████████    ░░███      █████  █████████       .  . ' ' .  . '     (__/  ")
    logger.info('░░░░░   ░░░░░  ░███░░░  ░░░░░  ░░░░░░░░      ░░░      ░░░░░  ░░░░░░░░░')
    logger.info('               ░███')                                                    
    logger.info('               █████')
    logger.info('               ░░░░░ \n')


    logger.info('Insect inspired vision system v0.0.1 \n')

    logger.info('© 2025 Adam D Hines¹⁻², Karin Nordström³, Andrew Barron¹ - <others TBD>')
    logger.info('Macquarie University¹, Queensland University of Technology², Flinders University³ \n')

    logger.info('MIT license - https://github.com/AdamDHines/ApiaViz')
    logger.info('Email - adam.hines@mq.edu.au \n')

    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        logger.info(f'CUDA available: {str(torch.cuda.is_available())} -- Current device is: {str(torch.cuda.get_device_name(current_device))}')
    elif torch.backends.mps.is_available():
        logger.info(f'MPS available: {str(torch.backends.mps.is_available())} -- Current device is: MPS')
    else:
        logger.info(f'CUDA available: {str(torch.cuda.is_available())} -- Current device is: CPU \n')
    model_label = args.snn_vision_model if args.snn else args.vision_model
    if getattr(args, "mode", None) == "train" and getattr(args, "train_stage", "backbone") == "lobula_plate":
        model_label = args.lobula_plate_model
    if getattr(args, "mode", None) == "train" and getattr(args, "train_stage", "backbone") == "projection":
        model_label = args.projection_model
    if getattr(args, "mode", None) == "train" and getattr(args, "train_stage", "backbone") == "reward_memory":
        model_label = args.reward_model
    if getattr(args, "mode", None) == "eval" and not args.snn:
        if getattr(args, "eval_feature", "kenyon_code") in {"reward_logit", "reward_probability"}:
            model_label = f"{args.lobula_plate_model} + {args.projection_model} + {args.reward_model}"
        else:
            model_label = f"{args.lobula_plate_model} + {args.projection_model}"

    if args.snn:
        logger.info(f'ApiaViz is running: Spiking neural network (SNN) mode with model {args.snn_vision_model}.pth \n')
    elif getattr(args, "mode", None) == "eval":
        if getattr(args, "eval_feature", "kenyon_code") in {"reward_logit", "reward_probability"}:
            logger.info(
                "ApiaViz is running: Artificial neural network (ANN) mode with checkpoints "
                f"{args.lobula_plate_model}.pth + {args.projection_model}.pth + {args.reward_model}.pth \n"
            )
        else:
            logger.info(
                "ApiaViz is running: Artificial neural network (ANN) mode with checkpoints "
                f"{args.lobula_plate_model}.pth + {args.projection_model}.pth \n"
            )
    else:
        logger.info(f'ApiaViz is running: Artificial neural network (ANN) mode with model {model_label}.pth \n')

    if args.mode == 'train':
        train_dataset_label = args.training_dataset
        if getattr(args, "train_stage", "backbone") == "reward_memory":
            train_dataset_label = args.reward_dataset
        logger.info('Training new model with the following parameters:')
        logger.info(f'  - Stage: {getattr(args, "train_stage", "backbone")}')
        logger.info(f'  - Dataset: {train_dataset_label}')
        logger.info(f'  - Epochs: {args.epochs}')
        logger.info(f'  - Batch size: {args.batch_size}')
        logger.info(f'  - Learning rate: {args.lr}')
        logger.info(f'  - Samples per image: {args.train_samples}')
        if getattr(args, "train_stage", "backbone") == "lobula_plate":
            logger.info(f'  - Spatial supervision: synthetic_shift')
            logger.info(f'  - Max shift: {args.spatial_max_shift}')
            logger.info(f'  - Min shift: {args.min_spatial_shift}')
            logger.info(f'  - Dense samples: {args.dense_samples}')
            logger.info(f'  - Dense temperature: {args.dense_temperature}')
            logger.info(f'  - Dense loss weight: {args.dense_loss_weight}')
            logger.info(f'  - Shift loss weight: {args.shift_loss_weight}')
            logger.info(f'  - Lobula LR scale: {args.lobula_lr_scale}')
            logger.info(f'  - Data loader workers: {args.num_workers}')
            logger.info(f'  - Init checkpoint: {args.backbone_checkpoint or "[auto-resolve latest pretrained backbone]"} \n')
        elif getattr(args, "train_stage", "backbone") == "projection":
            logger.info(f'  - Near shift range: {args.projection_near_min_shift} to {args.projection_near_max_shift}')
            logger.info(f'  - Far shift range: {args.projection_far_min_shift} to {args.projection_far_max_shift}')
            logger.info(f'  - VPN dim: {args.projection_vpn_dim}')
            logger.info(f'  - KC dim: {args.projection_kc_dim}')
            logger.info(f'  - Class grouping: {args.projection_class_grouping}')
            logger.info('  - Class objective: supervised contrastive')
            if getattr(args, "projection_kc_target_active", 0) > 0:
                effective_sparsity = max(
                    1.0 / float(args.projection_kc_dim),
                    float(args.projection_kc_target_active) / float(args.projection_kc_dim),
                )
                logger.info(f'  - KC target active count: {args.projection_kc_target_active}')
                logger.info(f'  - KC effective sparsity: {effective_sparsity:.4f}')
            else:
                logger.info(f'  - KC sparsity: {args.projection_kc_sparsity}')
            logger.info(
                '  - KC competition: '
                f'APL-like inhibition (gain={args.projection_apl_feedback_strength:.4f}, '
                f'adapt={args.projection_apl_gain_adapt_rate:.4f}, '
                f'threshold_lr={args.projection_apl_threshold_lr:.4f}, '
                f'iters={args.projection_apl_num_iters})'
            )
            logger.info(f'  - Feature loss weight: {args.projection_feature_loss_weight}')
            logger.info(f'  - Shift loss weight: {args.projection_shift_loss_weight}')
            logger.info(f'  - KC loss weight: {args.projection_kc_loss_weight}')
            logger.info(f'  - Class feature loss weight: {args.projection_class_feature_loss_weight}')
            logger.info(f'  - Class KC loss weight (peak): {args.projection_class_kc_loss_weight}')
            logger.info(
                f'  - Class KC curriculum: start epoch {args.projection_class_kc_start_epoch}, '
                f'ramp {args.projection_class_kc_ramp_epochs} epoch(s)'
            )
            logger.info(f'  - KC sparsity loss weight: {args.projection_kc_sparsity_loss_weight}')
            logger.info(f'  - Balance loss weight: {args.projection_balance_loss_weight}')
            logger.info(f'  - KC overlap loss weight: {getattr(args, "projection_kc_overlap_loss_weight", 0.0)}')
            logger.info(f'  - KC overlap margin: {getattr(args, "projection_kc_overlap_margin", 0.15)}')
            logger.info(
                f'  - KC negative overlap target: '
                f'{getattr(args, "projection_kc_negative_overlap_target", 0.10)}'
            )
            logger.info(f'  - KC usage loss weight: {getattr(args, "projection_kc_usage_loss_weight", 0.0)}')
            logger.info(f'  - Data loader workers: {args.num_workers}')
            logger.info(f'  - Init checkpoint: {args.backbone_checkpoint or args.lobula_plate_model} \n')
        elif getattr(args, "train_stage", "backbone") == "reward_memory":
            logger.info(f'  - Dataset: {args.reward_dataset}')
            logger.info(f'  - Reward feature: {args.reward_feature}')
            logger.info(f'  - Rewarded classes: {args.rewarded_classes or "[required at runtime]"}')
            logger.info(f'  - Reward head hidden dim: {args.reward_hidden_dim}')
            logger.info(f'  - Reward dropout: {args.reward_dropout}')
            logger.info(f'  - Reward val split: {args.reward_val_split}')
            logger.info(f'  - Reward threshold: {args.reward_threshold}')
            logger.info(f'  - Reward weight decay: {args.reward_weight_decay}')
            logger.info(f'  - Reward positive class weight: {args.reward_pos_weight}')
            logger.info(f'  - Data loader workers: {args.num_workers}')
            logger.info(
                f'  - Frozen checkpoints: '
                f'{args.backbone_checkpoint or args.lobula_plate_model} + '
                f'{args.projection_checkpoint or args.projection_model} \n'
            )
        else:
            logger.info('')
    elif args.mode == 'eval':
        logger.info('Evaluating model with the following parameters:')
        logger.info(f'  - Dataset: {args.eval_dataset}')
        logger.info(f'  - Samples per image: {args.eval_samples}')
        logger.info(f'  - Batch size: {args.eval_batch_size}')
        logger.info(f'  - Feature space: {getattr(args, "eval_feature", "kenyon_code")}')
        if getattr(args, "eval_feature", "kenyon_code") in {"reward_logit", "reward_probability"}:
            logger.info(f'  - Reward feature input: {args.reward_feature}')
            logger.info(f'  - Rewarded classes: {args.rewarded_classes or "[required at runtime]"}')
        logger.info(f'  - Scanning: {args.scanning} \n')

    return logger, output_folder
