# Compositional Sculpting of Iterative Generative Processes: Diffusion Models

This codebase provides an implementation of the diffusion model composition experiments on images. The diffusion model implementation is based on [Song et al.'s PyTorch tutorial notebook](https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3).

## Installation

Create a conda environment using the requirements file with the following command:
```bash
conda env create --name diffusion_sculpting --file environment.yml
conda activate diffusion_sculpting
```

## Code structure

* `models` folder contains implementations for the various score models and classifiers
* `samplers` contains an implementation of the PC sampler
* `custom_datasets.py` implements the various datasets used to train the base models
* `sample_individual_model.py` implements sample generation from one of the base models
* `sample_composition.py` implements sample generation from a composition of 2 base models
* `sample_3way_composition.py` implements sample generation from a composition of 3 base models
* `train_diffusion.py` implements the training procedure for the base models
* `train_classifier.py` implements the training procedure for training a classifier to sample from 2 base models
* `train_3way_classifier.py` trains a classifier for classifying the first two observations in a composition of 3 base models
* `train_3way_conditional_classifier.py` trains a classifier for classifying the third observation in a composition of 3 base models

## ColorMNIST experiment (3 base models)

This is the experimental setting reported in the main paper.

#### Base model training

This experiments composed 3 base models. The base models can be trained with: 
```bash
python3 -m train_diffusion
```
This scrips contains a variable `GEN_IDX` which determines which base model to train. Run the script three times with this variable set to `"MN1"`, `"MN2"` and `"MN3"` respectively.

#### Classifier training

In this setting we will be composing our three base models based on three observations. However, not all observations are classified by the same classifier. We will use two classifiers, one to classify the first two observations, and a second one to classify the last one conditioned on the first two.

Train the first classifier with
```bash
python3 -m train_3way_classifier.py
```

Once that has finished, train the second classifier with
```bash
python3 -m train_3way_conditional_classifier.py
```

#### Sampling from the composition

Once the classifiers have been trained we can generate samples from the resulting composition. To do this run
```bash
python3 -m sample_3way_composition.py
```
By default this generates samples from the composition correspondig to `y_1=1,y_2=2,y_3=3`. You can change this in the code by changing the definition of the composition. The composition is defined in two steps. First we construct a `BinaryDiffusionComposition` with the first two observations. Then we construct a `ConditionalDiffusionComposition` with the third observation.

## ColorMNIST and MNIST subdigits experiments (2 base models)

These two experiments are a simplification of the experiment above. Their results are discussed in the appendix of the paper.

#### Base model training

These experiments composed 2 base models. The base models can be trained with:
```bash
python3 -m train_classifier
```
This scrips contains a variable `GEN_IDX` which determines which base model to train. For the ColorMNIST instance set `GEN_IDX` to `"M1"` and `"M2"` respectively. For subdigits set `GEN_IDX` to `1` and `2` respectively.

#### Classifier training

In this setting we will be composing our base models based on two observations.

Train the classifier with
```bash
python3 -m train_3way_classifier.py
```
To train the classifier correctly we need to ensure that the correct score models are loaded. For ColorMNIST, please ensure that `score_model1` and `score_model2` are loaded from checkpoints `gen_M1_ckpt_195.pth` and `gen_M2_ckpt_195.pth` respectively. For subdigits, change these checkpoints to `gen_1_ckpt_195.pth` and `gen_2_ckpt_195.pth` respectively and set the `input_channels` variable in the ScoreNet constructors to `1`.

#### Sampling from the composition

Once the classifier have been trained we can generate samples from the resulting composition. To do this run
```bash
python3 -m sample_composition.py
```
Similar to before, the composition is defined by a `BinaryDiffusionComposition` object whose constructor takes the observations defining the composition as inputs. By default the code is set-up for the ColorMNIST condition. This means that `score_model1` and `score_model2` are loaded from checkpoints `gen_M1_ckpt_195.pth` and `gen_M2_ckpt_195.pth` respectively. For subdigits, change these checkpoints to `gen_1_ckpt_195.pth` and `gen_2_ckpt_195.pth` and set the `INPUT_CHANNELS` global variable to `1`.
