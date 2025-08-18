<p align="center">
  <img src="./assets/logo.png" alt="ApiaViz Logo" width="600"/>
</p>

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
[![Documentation Status](https://readthedocs.org/projects/apiaviz/badge/?version=latest&style=flat)](https://apiaviz.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)
[![GitHub repo size](https://img.shields.io/github/repo-size/AdamDHines/ApiaViz.svg?style=flat-square)](./README.md)

_TODO: Animation/GIF of ApiaViz demo_

This respository contains code for ApiaViz, a neural network model of insect vision using [Python](https://www.python.org/) and [PyTorch](https://pytorch.org/) for understanding natural scenes and environments. There are two versions of the model provided:

- An Artificial Neural Network (**ANN**) 
- A Spiking Neural Network (**SNN**) implemented in [SNNTorch](https://open-neuromorphic.org/neuromorphic-computing/software/snn-frameworks/snntorch/)

Get started easily by following our simple installation and quickstart instructions, below. _For more information, please visit the [ApiaViz documentation](https://apianet.readthedocs.io/en/latest/)._

## Installation
ApiaNet uses [pixi](https://prefix.dev/) to install and manage Python and dependencies. If you have not already installed it, run the following in your command terminal:

#### Linux/macOS
```console
curl -fsSL https://pixi.sh/install.sh | sh
```

#### Windows
```console
powershell -ExecutionPolicy ByPass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```

_For more information, please refer to the [pixi documentation](https://pixi.sh/latest/)._

### Get the code
Once installed, download the ApiaViz repository and navigate to the project directory:

```console
git clone git@github.com:AdamDHines/ApiaViz.git
cd ApiaViz
```

### Get the pre-trained models and evaluation datasets

We provide pre-trained models for the artificial and spiking versions of our vision model, as well as some evaluation datasets, from [Hugging Face](https://huggingface.co/). Run the following in your command terminal to get both:

```console
pixi run get_models
pixi run get_evaldata
```

## Quickstart

To run the evaluation, we can use the **pre-trained models and downloaded evaluation datasets** to assess the network quickly and easily:

```console
pixi run eval
pixi run eval -s
```
By default, the system will use the **ANN**. Using the `-s` argument will run the **SNN**.

To evaluate a different dataset, the `-d` argument can be used:

```console
pixi run eval -d variety
```
To change the evaluation method from the full image to a **scanning view**, we can use the `-sc` argument which will run a **smaller patch over the image** and temporally accumulate output:

```console
pixi run eval -sc
```
_For a full list of command line arguments, please visit the [ApiaViz documentation](https://apiaviz.readthedocs.io/en/latest/)._

## Using the vision model externally

To use the vision model in your experimental paradigm, we simply need to load the relevant model and pre-trained weights:

```python
from apiaviz.src.modules import VisionModel, SNNVisionModel

# ANN
visionmodel = VisionModel()
state_dict = torch.load('./apiaviz/models/VisionModel.pth', weights_only=True) # modify model path to your external program
visionmodel.load_state_dict(state_dict, strict=False)
visionmodel.eval()

# SNN
snnmodel = SNNVisionModel()
state_dict = torch.load('./apiaviz/models/SNNVisionModel.pth', weights_only=True) # modify model path to your external program
snnvisionmodel.load_state_dict(state_dict, strict=False)
snnvisionmodel.eval()
```
The **ANN** uses single image tensors of shape `[B, C, W, H]` whereas the **SNN** uses temporal sequences of image tensors of shape `[T, B, C, W, H]`, where `T` is the timesteps, `B` is the batch, `C` is the channel, and `W/H` is the width and height. Importantly, the vision systems only process **Blue** and **Green** channels, so if working with RGB images please ensure you are selecting the appropriate color channels:

```python
import torch

# Generate random input
ann_input = torch.randn(128, 2, 75, 75) # shape [B, C, W, H]
snn_input = torch.randn(25, 128, 2, 75, 75) # shape [T, B, C, W, H]

# Pass through corresponding model
KC_output_ann = visionmodel(ann_input)
KC_output_snn = snnvisionmodel(snn_input) 
```
This will return a sparse Kenyon Cell output of size `[B, 1024]` and `[T, B, 1024]` for the **ANN** and **SNN**, respectively.

For more details and a full guide, please visit the [ApiaViz documentation](https://apiaviz.readthedocs.io/en/latest/).

### Training new models
We provide pre-trained models for ApiaViz using the [Tiny ImageNet dataset](https://huggingface.co/datasets/zh-plus/tiny-imagenet), however if you would wish to train on another dataset or try different hyperparameters you can easily re-train the **ANN** and **SNN** models.

```console
# Optional: download the Tiny ImageNet dataset
pixi run get_tinyimg

# Run the training (CUDA enabled)
pixi run -e cuda train

# Train the SNN (CUDA enabled)
pixi run -e cuda train -s
```
_If not using `CUDA` as your device, a warning will be shown indicating that training will be very slow. Training the SNN requires a GPU device with a high amount of memory (>30GB) and is recommended to use a high performance computing (HPC) cluster._



For more information, please refer to the [ApiaViz documentation](https://apiaviz.readthedocs.io/en/latest/).

## License and citation
This code is licensed under the permissive [MIT license](./LICENSE). If you use our code, please cite our [paper]():

```
@article{,
      title={}, 
      author={},
      journal={},
      year={},
      volume={},
      number={},
      doi={},
      url={}, 
}
```

## Issues, bugs, and feature requests
If you encounter problems whilst running the code or if you have a suggestion for a feature or improvement, please report it as an [issue](https://github.com/AdamDHines/ApiaViz/issues).