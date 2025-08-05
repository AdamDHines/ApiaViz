# ApiaNet - Model honeybee active vision and action selection
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
[![Documentation Status](https://readthedocs.org/projects/apianet/badge/?version=latest&style=flat)](https://apianet.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)
[![GitHub repo size](https://img.shields.io/github/repo-size/AdamDHines/ApiaNet.svg?style=flat-square)](./README.md)

_TODO: Animation/GIF of ApiaNet demo_

This respository contains code for ApiaNet, a neural network model of honeybee active vision and active selection. ApiaNet is built using [PyTorch](https://pytorch.org/) and a modular neural network architecture to fuse vision, gustatory, and motor control for reward directed behavior.

Get started easily by following our simple installation and quickstart instructions, below.

_For more information, please visit the [ApiaNet documentation](https://apianet.readthedocs.io/en/latest/)._

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
Once installed, download the ApiaNet repository and navigate to the project directory:

```console
git clone git@github.com:AdamDHines/ApiaNet.git
cd ApiaNet
```

## Quickstart
### Run the demo
_TODO_

```console
pixi run demo
```

### Train and evaluate a new agent
_TODO_

```console
pixi run train
pixi run eval
```

For more information, please refer to the [ApiaNet documentation](https://apianet.readthedocs.io/en/latest/).

## Train modules
To train new modules, we can train them individually by running the following in the command terminal:

```console
pixi run train_vision
pixi run train_gustatory
pixi run train_motor
```

_Note: the `MotorModule` requires a pre-trained gustatory module before running._

For more information, please refer to the [ApiaNet documentation](https://apianet.readthedocs.io/en/latest/).

## Evaluate modules
Individual modules can be evaluated for their respective functions and accuracy. To evaluate the modules, run the following in your command terminal:

```console
pixi run eval_vision
pixi run eval_gustatory
pixi run eval_motor
```

For more information, please refer to the [ApiaNet documentation](https://apianet.readthedocs.io/en/latest/).

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
If you encounter problems whilst running the code or if you have a suggestion for a feature or improvement, please report it as an [issue](https://github.com/AdamDHines/ApiaNet/issues).