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
    if args.snn:
        logger.info(f'ApiaViz is running: Spiking neural network (SNN) mode with model {args.snn_vision_model}.pth \n')
    else:
        logger.info(f'ApiaViz is running: Artificial neural network (ANN) mode with model {args.vision_model}.pth \n')

    if args.mode == 'train':
        logger.info('Training new model with the following parameters:')
        logger.info(f'  - Dataset: Tiny ImageNet')
        logger.info(f'  - Epochs: {args.epochs}')
        logger.info(f'  - Batch size: {args.batch_size}')
        logger.info(f'  - Learning rate: {args.lr}')
        logger.info(f'  - Samples per image: {args.train_samples} \n')
    elif args.mode == 'eval':
        logger.info('Evaluating model with the following parameters:')
        logger.info(f'  - Dataset: {args.eval_dataset}')
        logger.info(f'  - Samples per image: {args.eval_samples}')
        logger.info(f'  - Batch size: {args.eval_batch_size}')
        logger.info(f'  - Scanning: {args.scanning} \n')

    return logger, output_folder