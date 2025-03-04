# IMPNet
This is a Pytorch-Lightning implementation of the paper "*Efficient Instance Motion-Aware Point Cloud Scene Prediction *" submitted to **IROS 2025**.

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
Download the SematicKitti data from the official [website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php).

We process the data in advance to speed up training. To prepare the dataset from the KITTI odometry dataset, set the value of ```GENERATE_FILES``` to true in ```config/parameters.yaml```. The environment variable ```PCF_DATA_RAW``` points to the directory containing the train/val/test sequences specified in the config file. It can be set with

```bash
export PCF_DATA_RAW=/path/to/kitti-odometry/dataset/sequences
```

and the destination of the processed files ```PCF_DATA_PROCESSED``` is set with

```bash
export PCF_DATA_PROCESSED=/desired/path/to/processed/data/
```

## Training
After following the [data preparation](#data-preparation) tutorial, the model can be trained in the following way:


The training script can be run by
```bash
python -m atppnet.train
```
using the parameters defined in ```config/parameters.yaml```. Pass the flag ```--help``` if you want to see more options like resuming from a checkpoint or initializing the weights from a pre-trained model. A directory will be created in ```pcf/runs``` which makes it easier to discriminate between different runs and to avoid overwriting existing logs. The script saves everything like the used config, logs and checkpoints into a path ```pcf/runs/COMMIT/EXPERIMENT_DATE_TIME``` consisting of the current git commit ID (this allows you to checkout at the last git commit used for training), the specified experiment ID (```pcf``` by default) and the date and time.

*Example:*
```pcf/runs/7f1f6d4/pcf_20211106_140014```

```7f1f6d4```: Git commit ID

```pcf_20211106_140014```: Experiment ID, date and time


## Testing
Test your model by running
```bash
python -m atppnet.test -m COMMIT/EXPERIMENT_DATE_TIME
```
where ```COMMIT/EXPERIMENT_DATE_TIME``` is the relative path to your model in ```pcf/runs```. *Note*: Use the flag ```-s``` if you want to save the predicted point clouds for visualiztion and ```-l``` if you want to test the model on a smaller amount of data.

*Example*
```bash
python -m atppnet.test -m 7f1f6d4/pcf_20211106_140014
```
or 
```bash
python -m atppnet.test -m 7f1f6d4/pcf_20211106_140014 -l 5 -s
```
if you want to test the model on 5 batches and save the resulting point clouds.



## Download
Please download the model file from (here)


## Acknowledgment
