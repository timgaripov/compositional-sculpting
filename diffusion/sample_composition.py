# the code here is mostly copied from this tutorial: https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing

import numpy as np
import torch
import functools
import torch
import torch.nn as nn
import tqdm

from models.score_model import ScoreNet
from models.classifier_model import JointYClassifier
from models.compositions import BinaryDiffusionComposition

from samplers.pc_sampler import pc_sampler

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def marginal_prob_std(t, sigma):
  """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

  Args:    
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.  
  
  Returns:
    The standard deviation.
  """    
  t = torch.tensor(t, device=device)
  return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(t, sigma):
  """Compute the diffusion coefficient of our SDE.

  Args:
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.
  
  Returns:
    The vector of diffusion coefficients.
  """
  return torch.tensor(sigma**t, device=device)
  
sigma =  25.0
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

#
# Sampling
#

INPUT_CHANNELS = 3

score_model1 = ScoreNet(marginal_prob_std=marginal_prob_std_fn, input_channels=INPUT_CHANNELS)
score_model1 = score_model1.to(device)
ckpt1 = torch.load('gen_M1_ckpt_195.pth', map_location=device)
score_model1.load_state_dict(ckpt1)
for param in score_model1.parameters():
    param.requires_grad = False

score_model2 = ScoreNet(marginal_prob_std=marginal_prob_std_fn, input_channels=INPUT_CHANNELS)
score_model2 = score_model2.to(device)
ckpt2 = torch.load('gen_M2_ckpt_195.pth', map_location=device)
score_model2.load_state_dict(ckpt2)
for param in score_model2.parameters():
    param.requires_grad = False

classifier = JointYClassifier(input_channels=INPUT_CHANNELS)
classifier = classifier.to(device)
cls_ckpt = torch.load('classifier_ckpt_200.pth', map_location=device)
classifier.load_state_dict(cls_ckpt)
for param in classifier.parameters():
    param.requires_grad = False

composed_model = BinaryDiffusionComposition([score_model1, score_model2], classifier, 1, 1, 20.0)

sample_batch_size = 64
num_steps = 500

## Generate samples using the specified sampler.
samples = pc_sampler(composed_model, 
                    marginal_prob_std_fn,
                    diffusion_coeff_fn, 
                    sample_batch_size, 
                    num_steps=num_steps,
                    device=device)

## Sample visualization.
samples = samples.clamp(0.0, 1.0)

from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def convert_colorblind(X):
    X = X.cpu()
    if X.shape[1] == 1:
       return X
    
    # colorblind_transform = torch.tensor([[0.83, 0.07, 0.35],[0.1, 0.52, 1.0], [0.0, 0.0, 0.0]])
    colorblind_transform = torch.tensor([[225/255, 190/255, 106/255],[64/255, 176/255, 166/255], [0.0, 0.0, 0.0]])
    Xcb = torch.zeros_like(X)
    for i in range(X.shape[0]):
       for x in range(X.shape[2]):
          for y in range(X.shape[3]):
             Xcb[i,:,x,y] = X[i,:,x,y] @ colorblind_transform
    return Xcb

sample_grid = make_grid(convert_colorblind(samples), nrow=int(np.sqrt(sample_batch_size)))

plt.figure(figsize=(6,6))
plt.axis('off')
plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
plt.show()