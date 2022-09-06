# Introduction

This folder contains the `Temporal Fushion Transformer` implemented in [`PytorchForecasting`](https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer.html) framework. It supports dual prediction (case and death). However, the framework is generic, so both single and multiple outputs are easy to implement.

## Folder structure
* **2022_May_cleaned**: Contains the merged feature files from raw dataset, which can be directly fed into the models. The `Total.csv` file isn't commited due to size limit. But can be reproduced locally using the [prepare_data.py](script/prepare_data.py). It also contains merged features for top 100 and 500 counties, which can be used for a quick run of the experiment.
* **Class**
  * `DataMerger.py`
  * `Parameters.py`
  * `PlotConfig.py`
  * `Plotter.py`
  * `PredictionProcessor.py`
* **configurations**: Configurations of the TFT model and data preprocessing saved in json format. They are used to reproduce the experiments.
* **results**: Results on the primary split using 3,142 US counties. For TFT it contains the final and best checkpoint, figures plotted using the results from the best validated model.
  * checkpoints
    * epoch=X-step=X.ckpt: model checkpointed by best validation loss.
    * model.ckpt: final model saved after finishing traning.
  * figures: saves the figures plotted by the final model obtained after finishing the training.
  * figures_best: figures plotted using the model with best validation loss. Used for the paper results.
  * lightning_logs: This folder is used by tensorboard to log the training and validation visualization. Not commited in git. You can point this folder by clicking the line before `import tensorboard as tb` in the training code (both script and notebook), that says `launch tensorboard session`. VSCode will automatically suggest the extensions needed for it. It can also run from cmd line, using `tensorboard --logdir=lightning_logs`, then it'll show something like `TensorBoard 2.9.0 at http://localhost:6006/ (Press CTRL+C to quit)`. Copy paste the URL in your local browser. To save the images, check `show data download links in the top left`.
* **results_split**: Results on the additonal splits using 3,142 US counties. Split 1, 2, 3 are respectively rising, falling and post 3rd wave splits.
* **script**: Contains scripts for submitting batch jobs. For details on how to use then, check the [README.md](script/README.md) inside the folder.
  * `prepare_data.py`: Prepare merged data from raw feature files.
  * `train.py`: Train model on merged data, then interpret using the best model by validation loss.
  * `inference.py`: Inference from a saved checkpoint.
  * `utils.py`: Contains utility methods.

## Configuration

This section describes how the configuration files work. The purpose of the configuration files are

* Record TFT model and experiment parameters
  * So hidden layer size, loss metric, epoch, learning rate all are supposed to be here.
* Provide data feature maps and the feature file locations.
  * If you want to add or remove features, static or dynamic add the feature to corresponding raw file mapping here.
  * This release can handle multiple features from a single file. E.g. you can replace `"Vaccination.csv": "VaccinationFull"` with `"Vaccination.csv": ["VaccinationFull", "VaccinationSingleDose"]` if  Or `Vaccination.csv` has feature columns for both of them.
* Paths and mapping for supporting files used, like `Population.csv`.
* Start and end dates for train, test, validation split.

## Environment

### Runtime

Currently on Rivanna with batch size 64, each epoch with

* Top 100 counties takes around 2-3 minutes.
* Top 500 counties takes around 12-13 minutes, memory 24GB.
* Total 3,142 counties takes around 40-45 minutes, memory 32GB.

### Google Colab

If you are running on **Google colab**, most libraries are already installed there. You'll only have to install the pytorch forecasting and lightning module. Uncomment those installation commands in the code. Upload the TFT-pytorch folder in your drive and set that path in the notebook colab section. If you want to run the data preparation notebook, upload the [CovidMay17-2022](../dataset_raw/CovidMay17-2022/) folder too. Modify the path accordingly in the notebook.

```python
!pip install pytorch_lightning
!pip install pytorch_forecasting
```

### Rivanna/CS server

On **Rivanna**, the default python environment doesn't have all the libraries we need. The [requirements.txt](requirements.txt) file contains a list of libraries we need. There are two ways you can run the training there

#### Default Environment

Rivanna provides a bunch of python kernels readily available. You can check them from an interactive Jupyterlab session, on the top-right side of the notebook. I have tested with the `Tensorflow 2.8.0/Keras Py3.9` kernel and uncommented the following snippet in the code.

```python
!pip install pytorch_lightning
!pip install pytorch_forecasting
```

You can choose different kernels and install the additional libraries. 

#### Virtual Environment

You can directly create a python virtual environment using the [environment.yml](environment.yml) file and Anaconda. Then you won't have to install the libraries each time. Copy this file to your home directory and run the following command,

```bash
conda create --name <env> --file <this file>

# for example
conda create --name ml --file environment.yml

# then activate the environment with
conda activate ml
# now you should be able to run the files from your cmd line without error
# if you are on notebook select this environment as your kernel
```

You can also create the environment in this current directory, then the virtual environment will be saved in this folder instead, not in the home directory.

#### GPU 

Next, you might face issues getting GPU running on Rivanna. Even on a GPU server the code might not recognize the GPU hardware if cuda and cudnn are not properly setup. Try to log into an interactive session in a GPU server, then run the following command

```bash
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```

If this is still 0, then you'll have to install the cuda and cudnn versions that match version in `nvidia-smi` command output. Also see if you tensorflow version is for CPU or GPU.

## How to replicate

* Write down your intended configuration in the config.json file. Or reuse an existing one.
  * Change TFT related parameters in the `"model_parameters"` section.
  * To train all 60 epochs change `"early_stopping_patience"` to 60. Default is `3`.
  * To add/remove new features or for different population cut use the `"data"` section.
  * `Data` section also has the train/validation/test split.
  * Use the `preprocess` section to remove outliers during the data preparation.
  * **Note**: Changing the order or spelling of the json keys will require chaning the [Parameters.py](/Class/Parameters.py) accordingly.
  
* Use the data prepration [script](/script/prepare_data.py) to create the merged data.
  * Make sure to pass the correct config.json file.
  * Check the folder paths in `args` class, whether they are consistent.
  * Depending on your configuration it can create the merged file of all counties, based on a population cut (e.g. top 500 counties) or rurality cut. All counties are saved in `Total.csv`, population cut in `Top_X.csv` where `X` is the number of top counties by population. Rurality is saved in `Rurality_cut.csv`.
  * Currently there is a option to either remove outliers from the input and target, or not. Removing target outliers can decrease anomalies in the learning. But changing the ground truth like this is not often desirable, so you can set it to false in the `preprocess` section in the configuration.
  * Note that, scaling and splitting are not done here, but later during training and infering.
  
* Use the training [script](/script/train.py) to train and interpret the model.
  * Make sure to pass the correct config.json file.
  * Check the folder paths in `args` class, whether they are consistent.
  * This file reads the merged feature file, splits it into train/validation/test, scales the input and target if needed, then passes them to the model.
  * The interpretation is done for both the final model and the model with best validation loss.
  * Note the path where the models are checkpointed.
  * Using vscode you can also open the tensorboard to review the training logs.
  * The prediction is saved as csv file for any future visualizations.
  
* The inference [script](/script/inference.py) can be used to infer a previously checkpointed model and interpret it.
  * Same as before, recheck the config.json file and `args` class. Make sure they are same as the model you are going to infer.
  * Set the model path to the checkpoint model you want to infer.

## Usage guideline

* Please do not add temporarily generated files in this repository.
* Make sure to clean your tmp files before pushing any commits.
* In the .gitignore file you will find some paths in this directory are excluded from git tracking. So if you create anything in those folders, they won't be tracked by git.
  * To check which files git says untracked `git status -u`. 
  * If you have folders you want to exclude add the path in `.gitignore`, then `git add .gitignore`. Check again with `git status -u` if it is still being tracked.
