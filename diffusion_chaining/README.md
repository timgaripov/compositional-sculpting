#

## Setup

We use **[params-proto](https://pypi.org/project/params-proto)** to specify the hyperparameters, and generate [sweep.jsonl](experiments/sculpting.jsonl) files. We use **[ml-logger](https://pypi.org/project/ml-logger)** to centralize metrics
logging and checkpointing. We use **[jaynes](https://pypi.org/project/jaynes)** to launch the experiments on the cloud.

```
conda create -n sculpting python=3.8
conda install pycurl
pip install params-proto jaynes ml-logger cloudpickle==1.3.0
```

## Colored MNIST Experiments

There are two main experiments: Binary composition between two base diffusion models, and chained composition between three base
diffusion models. The steps are:
1. train base diffusion models on various colored MNIST digit distributions
2. train binary classifier over pairs
3. sample from the binary compositions
4. train binary classifier over pairs used in the chaining experiments
5. sample from the chained compositions

The **[experiments](experiments)** folder contains
the scripts used to launch the training and sampling. The **[models](models)** folder contains the model definitions.

```
diffusion_chaining
├── README.md
├── __init__.py
├── experiments
│   ├── chain.jsonl
│   ├── chain.py
│   ├── ddpm.jsonl
│   ├── ddpm.py
│   ├── sculpting.jsonl
│   └── sculpting.py
├── bcomp.py
├── bcomp_sampler.py
├── chain.py
├── chain_sampler.py
├── ddpm.py
├── ddpm_sampler.py
└── models
    ├── classifier_model.py
    ├── score_model.py
    └── util.py
3 directories, 17 files
```

### Preparation: Training Base Diffusion Models

```python
ddpm.py
```

and to sample from these models:

```python
ddpm_sampler.py
```

### Experiment I: Binary Composition: Training Binary Classifier

```python
bcomp.py
```

now, to sample from the binary compositions

```python
bcomp_sampler.py
```

### Experiment II: Training Chained Classifiers

```python
chain.py
```
now to sample from the chained compositions

run:
```python
python chain_sampler.py
```
