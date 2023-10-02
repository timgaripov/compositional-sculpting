# Compositional Sculpting of Iterative Generative Processes: GFlowNets

This codebase provides implementation of GFlowNet experiments in two domains:

* 2D drid
* Fragment-based molecule generation

## Installation and requirements

Create conda environment using the requirements file with the following command:
```bash
# In the root directory of the repository
conda env create --name gflownet_sculpting --file ./gflownet/environment.yml
conda activate gflownet_sculpting
```

This codebase relies on the following experiment pipeline tools:

* [params-proto](https://github.com/geyang/params_proto) -- tool for managing hyperparameters and sweeps.
* [Jaynes](https://github.com/geyang/jaynes) -- tool for running experiments on various compute platforms such as:
  * servers accessible via SSH
  * HPC clusters with slurm
  * Cloud compute (GCP, AWS)
* [ml-logger](https://github.com/geyang/ml_logger) -- tool for logging metrics and artifacts.
* [cmx-python](https://github.com/cmx/cmx-python) -- tool for generating markdown reports in python. 

These tools are available as python packages through pip.

Resources:
* params-proto source code: https://github.com/geyang/params_proto
* Jaynes source code: https://github.com/geyang/jaynes
* Jaynes starter kit: https://github.com/geyang/jaynes-starter-kit
* ml-logger source code: https://github.com/geyang/ml_logger
* cmx-python source code: https://github.com/cmx/cmx-python


Set the environment variables for ml-logger logs destination. For example, to save reuslts locally, run:
```bash
# Root directory for the logs
export ML_LOGGER_ROOT=/tmp/gflownet_sculpting
# ml-logger user name
export ML_LOGGER_USER=sculptor 
```
See [ml-logger documentation](https://github.com/geyang/ml_logger) for more details and instructions for online logging. 


## 2D Grid

The implementation of GFlowNets for 2D grid domain is based on https://gist.github.com/malkin1729/9a87ce4f19acdc2c24225782a8b81c15.


### Code structure and example usage

* `grid` folder contains training scripts for GFlowNets and classifiers
* `experiment_scripts/grid` folder contains [jaynes](https://github.com/geyang/jaynes) launcher scripts for 2D grid experiments
* `experiment_analysis/grid` folder contains experiment result analysis scripts and generated figures

Each script in `grid` specifies a set of hypeparameters and default values for these parametes. The launcher scripts set individual hyperparameter values (or hyperparameter sweeps) for specific experiments.
The experiments can be run with the launcher scripts using jaynes or by running the scripts directly with appropriate hyperparameter values.

#### GFlowNet training

GFlowNet 2D grid training script: `gflownet/grid/train_grid.py`

The script can be run directly with the following command 
```bash
# In the root directory of the repository
python3 -m gflownet.grid.train_grid
```

Training hyperparameters can be passed as command line arguments.

ml-logger logs and artifacts will be saved in a subdirectory of `ML_LOGGER_ROOT` directory. The path to the experiment directory will be printed in the console output. It will look something like this
```bash
# Assuming ML_LOGGER_ROOT=/tmp/gflownet_sculpting and ML_LOGGER_USER=sculptor
/tmp/gflownet_sculpting/sculptor/scratch/2023/09-28/gflownet/grid/train_grid/11.29.19/1/
```

Training of base GFlowNets in 2D grid domain for the experiments in the paper can be reproduced by running the jaynes launcher scripts:

* `gflownet/experiment_scripts/grid/run_grid_2dist.py` -- 2 distributions experiment
* `gflownet/experiment_scripts/grid/run_grid_3dist.py` -- 3 distributions experiment 

Note that the launcher scripts are configured to run the code locally and to run only a single trial. If you need to change these settings, edit the launcher scripts. 
See [jaynes documentation](https://github.com/geyang/jaynes) and [jaynes starter kit](https://github.com/geyang/jaynes-starter-kit) for more details.

#### Classifier training

Classifier training scripts:

* `gflownet/grid/train_grid_cls_2dist_param.py` -- classifier training for 2 models composition
* `gflownet/grid/train_grid_cls_3dist.py` -- classifier training for 3 models composition 

Paths to the pre-trained GFlowNets need to be provided as arguments to the scripts (either as command line arguments or in the launcher scripts).

Example direct execution command:
```bash
# In the root directory of the repository
python3 -m gflownet.grid.train_grid_cls_2dist_param \
  --classifier-2dist.run-path-1=sculptor/scratch/2023/04-18/run_grid/13.40.18/symmetric_shubert_uniform_pb_100 \
  --classifier-2dist.run-path-2=sculptor/scratch/2023/04-18/run_grid/13.40.18/diag_sigmoid_uniform_pb_100
```
Note that the GFlowNet training run paths are relative to the `ML_LOGGER_ROOT` directory. The exact paths might be different depening on the way the base GFlowNet training scripts were run.

Similarly, the classifier training for 3 models composition can be run with the following command template:
```bash
# In the root directory of the repository
python3 -m gflownet.grid.train_grid_cls_3dist \
  --classifier-3dist.run-path-1=sculptor/scratch/2023/04-18/run_grid/15.12.33/circle1 \
  --classifier-3dist.run-path-2=sculptor/scratch/2023/04-18/run_grid/15.12.33/circle2 \
  --classifier-3dist.run-path-3=sculptor/scratch/2023/04-18/run_grid/15.12.33/circle3
```

Training of classifiers in 2D grid domain for the experiments in the paper can be reproduced by running the jaynes launcher scripts:

* `gflownet/experiment_scripts/grid/run_grid_cls_2dist_param.py` -- 2 distributions experiment 
* `gflownet/experiment_scripts/grid/run_grid_cls_3dist.py` -- 3 distributions experiment

Note that the paths to the pre-trained GFlowNets need to be set in the launcher scripts.

#### Analysis of learned distributions

The evaluation and visualization of base distributions and composed distributions is done with the following analysis scripts:

* `gflownet/experiment_analysis/grid/grid_2dist_figures.py` -- analysis of 2 distributions experiment
* `gflownet/experiment_analysis/grid/grid_3dist_figures.py` -- analysis of 3 distributions experiment

Note that the paths to the pre-trained GFlowNets and classifiers need to be set in the analysis scripts.

## Fragment-based molecule generation

The implementation of GFlowNets for fragment-based molecule generation is based on https://github.com/recursionpharma/gflownet (last synced on 2023-04-06, with commit [3d311c3](https://github.com/recursionpharma/gflownet/tree/3d311c3d0d18f3c7f9ce67dd1fff502c44ca57d9)).

### Code structure and example usage

* `fragment` folder contains training scripts for GFlowNets and classifiers as well as GFlowNet evaluation scripts
* `experiment_scripts/fragment` folder contains [jaynes](https://github.com/geyang/jaynes) launcher scripts for molecule generation experiments
* `experiment_analysis/fragment` folder contains experiment result analysis scripts and generated figures

#### GFlowNet training
GFlowNet training script: `gflownet/fragment/mogfn.py`

The script can be run directly with the following command
```bash
# In the root directory of the repository
python3 -m gflownet.fragment.mogfn
```

The script trains a conditional GFlowNet model that learns the distributions $p_\beta(x) \propto (R(x))^\beta$ for a range of values of $\beta$. 
The model evaluation and classifier training scripts (see below) accept the value of $\beta$ as a parameter and use the policy corresponding to the specified $\beta$ value.

Training hyperparameters can be passed as command line arguments.

In our experiments each model was trained with a single GPU, the GPU id can be specified through the `CUDA_VISIBLE_DEVICES` environment variable.


ml-logger logs and artifacts will be saved in a subdirectory of `ML_LOGGER_ROOT` directory. The path to the experiment directory will be printed in the console output. It will look something like this
```bash
# Assuming ML_LOGGER_ROOT=/tmp/gflownet_sculpting and ML_LOGGER_USER=sculptor
/tmp/gflownet_sculpting/sculptor/scratch/2023/09-29/gflownet/fragment/mogfn/14.31.53/1
```

Training of base GFlowNets in molecule generation domain for the experiments in the paper can be reproduced by running the jaynes launcher script:

`gflownet/experiment_scripts/fragment/run_1obj_beta_cond.py`

Note that the launcher script is configured to run the code locally and to run only a single trial. If you need to change these settings, edit the launcher script.
See [jaynes documentation](https://github.com/geyang/jaynes) and [jaynes starter kit](https://github.com/geyang/jaynes-starter-kit) for more details.

#### GFlowNet policy evaluation

The script `gflownet/fragment/eval_model_beta.py` can be used to evaluate a trained GFlowNet policy. The script takes in a path to a GFlowNet training run, loads the trained model, and generates molecules using the policy. The generated molecules and their property scores (rewards) are saved in the ml-logger logs directory. 

Example direct execution command:
```bash
# In the root directory of the repository
python3 -m gflownet.fragment.eval_model_beta \
  --eval.model-path=sculptor/scratch/2023/09-29/gflownet/fragment/mogfn/14.31.53/1 \
  --eval.beta=32.0
```

`gflownet/experiment_scripts/fragment/run_eval_model_beta.py` is the jaynes launcher script for GFlowNet policy evaluation. 
Note that the paths to the pre-trained GFlowNets need to be set in the launcher script.

#### Classifier training

Classifier training scripts:

* `gflownet/fragment/train_joint_cls_param_onestep_beta.py` -- classifier training for 2 models composition
* `gflownet/fragment/train_3joint_cls_onestep_beta.py` -- classifier training for 3 models composition 


Paths to the pre-trained GFlowNets need to be provided as arguments to the scripts (either as command line arguments or in the launcher scripts).

Example direct execution commands:
```bash
# In the root directory of the repository
python3 -m gflownet.fragment.train_joint_cls_param_onestep_beta \
  --classifier-2dist.run-path-1=sculptor/scratch/2023/09-29/gflownet/fragment/mogfn/14.31.53/1 \
  --classifier-2dist.beta-1=32.0 \
  --classifier-2dist.run-path-2=sculptor/scratch/2023/09-29/gflownet/fragment/mogfn/14.38.10/1 \
  --classifier-2dist.beta-2=32.0
```

```bash
# In the root directory of the repository
python3 -m gflownet.fragment.train_3joint_cls_onestep_beta \
  --classifier-3dist.run-path-1=sculptor/scratch/2023/09-29/gflownet/fragment/mogfn/14.31.53/1 \
  --classifier-3dist.beta-1=32.0 \
  --classifier-3dist.run-path-2=sculptor/scratch/2023/09-29/gflownet/fragment/mogfn/14.38.10/1 \
  --classifier-3dist.beta-2=32.0 \
  --classifier-3dist.run-path-3=sculptor/scratch/2023/09-29/gflownet/fragment/mogfn/14.41.22/1 \
  --classifier-3dist.beta-3=32.0
```

Training of classifiers in the molecule generation domain for the experiments in the paper can be reproduced by running the jaynes launcher scripts:

* `gflownet/experiment_scripts/fragment/run_frag_joint_cls_param_onestep_beta.py` -- 2 distributions experiment
* `gflownet/experiment_scripts/fragment/run_frag_3joint_cls_onestep_beta.py` -- 3 distributions experiment

Note that the paths to the pre-trained GFlowNets need to be set in the launcher scripts.

#### Guided sampling policy evaluation

Classifier-guided sampling policy evaluation scripts:

* `gflownet/fragment/eval_model_guided_joint_param_beta.py` -- evaluation of 2 models composition
* `gflownet/fragment/eval_model_guided_3joint_beta.py` -- evaluation of 3 models composition

The scripts take in paths to the GFlowNets and classifier training runs, load the trained models, and generate molecules using the guided policy. The generated molecules and their property scores (rewards) are saved in the ml-logger logs directory. 

Example direct execution commands:
```bash
# In the root directory of the repository
python3 -m gflownet.fragment.eval_model_guided_joint_param_beta \
  --eval.model-path-1=sculptor/scratch/2023/09-29/gflownet/fragment/mogfn/14.31.53/1 \
  --eval.beta-1=32.0 \ 
  --eval.model-path-2=sculptor/scratch/2023/09-29/gflownet/fragment/mogfn/14.38.10/1 \
  --eval.beta-2=32.0 \
  --eval.cls-path=sculptor/scratch/2023/09-29/gflownet/fragment/train_joint_cls_param_onestep_beta/21.18.13/1 \
  --eval.cls-y1=1 --eval.cls-y2=2 
```

```bash
# In the root directory of the repository
python3 -m gflownet.fragment.eval_model_guided_3joint_beta \
  --eval.model-path-1=sculptor/scratch/2023/09-29/gflownet/fragment/mogfn/14.31.53/1 \
  --eval.beta-1=32.0 \ 
  --eval.model-path-2=sculptor/scratch/2023/09-29/gflownet/fragment/mogfn/14.38.10/1 \
  --eval.beta-2=32.0 \
  --eval.model-path-3=sculptor/scratch/2023/09-29/gflownet/fragment/mogfn/14.41.22/1 \
  --eval.beta-3=32.0 \
  --eval.cls-path=sculptor/scratch/2023/09-29/gflownet/fragment/train_3joint_cls_onestep_beta/21.30.14/1 \
  --eval.cls-y1=1 --eval.cls-y2=2  --eval.cls-y3=3
```

Jaynes launcher scripts for GFlowNet policy evaluation:

* `gflownet/experiment_scripts/fragment/run_eval_model_guided_joint_param_beta.py` -- 2 distributions experiment
* `gflownet/experiment_scripts/fragment/run_eval_model_guided_3joint_beta.py` -- 3 distributions experiment


Note that the paths to the pre-trained GFlowNets and classifiers need to be set in the launcher script.


#### Analysis of generated molecules

The evaluation and visualization of properties of generated molecules is done with the following analysis scripts:

* `gflownet/experiment_analysis/fragment/composition2_indep_param_beta32.py` -- analysis of 2 distributions compositions (beta=32)
* `gflownet/experiment_analysis/fragment/composition2_indep_param_beta96.py` -- analysis of 2 distributions compositions (beta=96)
* `gflownet/experiment_analysis/fragment/composition3_indep_beta32.py` -- analysis of 3 distributions compositions (beta=32)


Note that the paths to the pre-trained GFlowNets and classifiers need to be set in the analysis scripts.
