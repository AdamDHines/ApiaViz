# ApiaViz - A neural network model of the honeybee _Apis mellifera_ visual system
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
[![Documentation Status](https://readthedocs.org/projects/apiaviz/badge/?version=latest&style=flat)](https://apiaviz.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)
[![GitHub repo size](https://img.shields.io/github/repo-size/AdamDHines/ApiaViz.svg?style=flat-square)](./README.md)

_TODO: Animation/GIF of ApiaViz demo_

This respository contains code for ApiaViz, a neural network model of honeybee vision. ApiaViz is built using [PyTorch](https://pytorch.org/).

Get started easily by following our simple installation and quickstart instructions, below.

_For more information, please visit the [ApiaViz documentation](https://apianet.readthedocs.io/en/latest/)._

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
For more details and a full guide, please visit the [ApiaViz documentation](https://apiaviz.readthedocs.io/en/latest/).

### Train and evaluate a new agent
_TODO_

```console
pixi run train
pixi run eval
```

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