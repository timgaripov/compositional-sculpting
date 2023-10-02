# the code here is mostly copied from this tutorial: https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing

import numpy as np
import torch
import functools
from samplers.pc_sampler import pc_sampler

from models.score_model import ScoreNet

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

score_model = ScoreNet(marginal_prob_std=marginal_prob_std_fn, input_channels=3)
score_model = score_model.to(device)

## Load the pre-trained checkpoint from disk.
ckpt = torch.load('gen_MN1_ckpt_195.pth', map_location=device)
score_model.load_state_dict(ckpt)
for param in score_model.parameters():
    param.requires_grad = False

num_steps =  500
sample_batch_size = 64

## Generate samples using the specified sampler.
samples = pc_sampler(score_model, 
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
    
    #colorblind_transform = torch.tensor([[0.83, 0.07, 0.35],[0.1, 0.52, 1.0], [0.0, 0.0, 0.0]])
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