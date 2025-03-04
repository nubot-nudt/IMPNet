# IMPNet
This is a Pytorch-Lightning implementation of the paper "**Efficient Instance Motion-Aware Point Cloud Scene Prediction **" submitted to **IROS 2025**.

![](docs/ICRA_Architecture.png)
**ATPPNet Architecture.** **


## Table of Contents
1. [Installation](#Installation)
2. [Data](#Data)
3. [Training](#Training)
4. [Testing](#Testing)
5. [Download](#Dwnload)
6. [Acknowledgment](#Acknowledgment)


## Installation

Clone this repository and run 
```bash
cd impnet
git submodule update --init
```
to install the Chamfer distance submodule. The Chamfer distance submodule is originally taken from [here](https://github.com/chrdiller/pyTorchChamferDistance) with some modifications to use it as a submodule. All parameters are stored in ```config/parameters.yaml```.

In our project, all our dependencies are managed by miniconda. 
Use the following command to create the conda environment:

```conda env create -f impnet.yml```

Then activate the environment using the command ```conda activate atppnet```

## Data
Download the SemanticKITTI data from the official [website](http://semantic-kitti.org/).

We process the data in advance to speed up training. 
To prepare the dataset from the our dataset, set the value of ```GENERATE_FILES``` to true in ```config/parameters.yaml```.


## Training
After following the [data preparation](#data-preparation) tutorial, the model can be trained in the following way:


The training script can be run by
```bash
python train.py
```
using the parameters defined in ```config/parameters.yaml```. Pass the flag ```--help``` if you want to see more options like resuming from a checkpoint or initializing the weights from a pre-trained model. A directory will be created in ```pcf/runs``` which makes it easier to discriminate between different runs and to avoid overwriting existing logs. The script saves everything like the used config, logs and checkpoints into a path ```pcf/runs/COMMIT/EXPERIMENT_DATE_TIME``` consisting of the current git commit ID (this allows you to checkout at the last git commit used for training), the specified experiment ID (```pcf``` by default) and the date and time.

## Testing
Test your model by running
```bash
python test.py -m COMMIT/EXPERIMENT_DATE_TIME
```


## Download
Please download the model file from (here)


## Acknowledgment
