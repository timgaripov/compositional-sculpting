# the code here is mostly copied from this tutorial: https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing

import numpy as np
import torch
import functools
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import tqdm

from models.score_model import ScoreNet
from custom_datasets import *

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

sigma =  25.0
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)

def loss_fn(model, x, marginal_prob_std, eps=1e-5):
  """The loss function for training score-based generative models.

  Args:
    model: A PyTorch model instance that represents a 
      time-dependent score-based model.
    x: A mini-batch of training data.    
    marginal_prob_std: A function that gives the standard deviation of 
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
  """
  random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
  z = torch.randn_like(x)
  std = marginal_prob_std(random_t)
  perturbed_x = x + z * std[:, None, None, None]
  score = model(perturbed_x, random_t)
  loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
  return loss

#
# Hyperparams
#

n_epochs = 200
## size of a mini-batch
batch_size =  32
## beginning learning rate
lr_start=1e-2
## end learning rate
lr_end=1e-4

#
# Dataset & Model
#

GEN_IDX = "MN1"

dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
if GEN_IDX == 1:
    subdataset = Subset(dataset, np.argwhere(dataset.targets.numpy() < 6).flatten())
    n_input_channels = 1
elif GEN_IDX == 2:
    subdataset = Subset(dataset, np.argwhere(dataset.targets.numpy() > 3).flatten())
    n_input_channels = 1
elif GEN_IDX == "M1":
    subdataset = M1(root='.', train=True, download=True, transform=transforms.ToTensor())
    n_input_channels = 3
elif GEN_IDX == "M2":
    subdataset = M2(root='.', train=True, download=True, transform=transforms.ToTensor())
    n_input_channels = 3
elif GEN_IDX == "MN1":
    subdataset = MN1(root='.', train=True, download=True, transform=transforms.ToTensor())
    n_input_channels = 3
elif GEN_IDX == "MN2":
    subdataset = MN2(root='.', train=True, download=True, transform=transforms.ToTensor())
    n_input_channels = 3
elif GEN_IDX == "MN3":
    subdataset = MN3(root='.', train=True, download=True, transform=transforms.ToTensor())
    n_input_channels = 3
else:
   raise NotImplementedError
data_loader = DataLoader(subdataset, batch_size=batch_size, shuffle=True)

score_model = ScoreNet(marginal_prob_std=marginal_prob_std_fn, input_channels=n_input_channels)
score_model = score_model.to(device)

#
# Training
#

optimizer = Adam(score_model.parameters(), lr=lr_start)
scheduler = ExponentialLR(optimizer, np.exp(np.log(lr_end / lr_start) / n_epochs))
for epoch in (tqdm_epoch := tqdm.tqdm(range(n_epochs))):
    avg_loss = 0.
    num_items = 0
    for x, y in data_loader:
        x = x.to(device)
        loss = loss_fn(score_model, x, marginal_prob_std_fn)
        optimizer.zero_grad()
        loss.backward()    
        optimizer.step()
        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]
    scheduler.step()
    # Print the averaged training loss so far.
    tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
    # Update the checkpoint after each epoch of training.
    if (epoch % 5 == 0) and (epoch != 0):
        torch.save(score_model.state_dict(), 'gen_' + str(GEN_IDX) + '_ckpt_' + str(epoch) + '.pth')