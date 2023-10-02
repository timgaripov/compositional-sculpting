# the code here is mostly copied from this tutorial: https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing

import functools

import numpy as np
import torch
import torchvision.transforms as transforms
import tqdm
from ml_logger.job import RUN
from params_proto import PrefixProto, Proto
from params_proto.hyper import Sweep
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from diffusion_chaining.models.score_model import ScoreNet


class DDPM(PrefixProto, cli=False):
    data_dir = Proto(env="$DATASETS")
    dataset = "m1"

    in_dim = 3
    sigma = 25.0

    # training params
    n_epochs = 200
    batch_size = 32
    lr_0 = 1e-2
    lr_T = 1e-4
    cp_interval = 50

    # sampling parameters
    n_steps = 200

    seed = 100

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")


def diffusion_coeff(t, sigma=DDPM.sigma, device: torch.DeviceObjType = DDPM.device):
    """Compute the diffusion coefficient of our SDE.

    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.

    Returns:
      The vector of diffusion coefficients.
    """
    return sigma**t


def marginal_prob_std(t, sigma):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.

    Returns:
      The standard deviation.
    """
    t = torch.tensor(t, device=DDPM.device)
    return torch.sqrt((sigma ** (2 * t) - 1.0) / 2.0 / np.log(sigma))


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
    random_t = torch.rand(x.shape[0], device=x.device) * (1.0 - eps) + eps
    z = torch.randn_like(x)
    std = marginal_prob_std(random_t)
    perturbed_x = x + z * std[:, None, None, None]
    score = model(perturbed_x, random_t)
    loss = torch.mean(torch.sum((score * std[:, None, None, None] + z) ** 2, dim=(1, 2, 3)))
    return loss


def get_dataset(key):
    from img_diffusion import ge_distribution

    Cls = getattr(ge_distribution, key.upper())

    dataset = Cls(root=DDPM.data_dir, train=True, download=True, transform=transforms.ToTensor())
    data_loader = DataLoader(dataset, batch_size=DDPM.batch_size, shuffle=True)
    return data_loader


def main(**deps):
    from ml_logger import logger

    DDPM._update(deps)
    logger.log_params(DDPM=vars(DDPM))
    print(logger.get_dash_url())
    # fmt: off
    logger.log_text("""
    charts:
    - yKey: loss/mean
      xKey: epoch
    - yKey: lr/mean
      xKey: epoch
    """, ".charts.yml", True, True)
    # fmt: on

    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=DDPM.sigma)
    score_model = ScoreNet(marginal_prob_std=marginal_prob_std_fn, input_channels=DDPM.in_dim)
    score_model = score_model.to(DDPM.device)

    data_loader = get_dataset(DDPM.dataset)
    optimizer = Adam(score_model.parameters(), lr=DDPM.lr_0)
    scheduler = ExponentialLR(optimizer, np.exp(np.log(DDPM.lr_T / DDPM.lr_0) / DDPM.n_epochs))

    for epoch in (tqdm_epoch := tqdm.tqdm(range(DDPM.n_epochs + 1))):
        if epoch and epoch % DDPM.cp_interval == 0:
            logger.torch_save(score_model, f"checkpoints/model_{epoch:03d}.pt")
            logger.duplicate(f"checkpoints/model_{epoch:03d}.pt", f"checkpoints/model_last.pt")

        if epoch == DDPM.n_epochs:
            break

        for x, y in data_loader:
            x = x.to(DDPM.device)

            loss = loss_fn(score_model, x, marginal_prob_std_fn)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logger.store_metrics(
                loss=loss.item(),
                lr=scheduler.get_last_lr()[0],
            )

        scheduler.step()
        # Print the averaged training loss so far.
        # tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        logger.log_metrics_summary(key_values={"epoch": epoch})
        # print('Average Loss: {:5f}'.format(avg_loss / num_items))
        # Update the checkpoint after each epoch of training.

    from diffusion_chaining.ddpm_sampler import collect_images

    collect_images(score_model, f"figures/{DDPM.dataset}.png")
    # fmt: off
    logger.log_text("""
    - type: image
      glob: "**/*.png"
    """, ".charts.yml", dedent=True)


if RUN.debug and __name__ == "__main__":
    from ml_logger.job import instr

    thunk = instr(main)

    thunk(**{"DDPM.dataset": "m1"})
    thunk(**{"DDPM.dataset": "m2"})

if __name__ == "__main__":
    import jaynes

    from ml_logger.job import instr

    jaynes.config("local")
    sweep = Sweep(DDPM, RUN).load("analysis/sweeps/ddpm.jsonl")

    gpus_to_use = [0, 1, 2, 3]

    for i, deps in enumerate(sweep):

        RUN.CUDA_VISIBLE_DEVICES = str(gpus_to_use[i % len(gpus_to_use)])
        jaynes.config("local")
        thunk = instr(main, **deps, __diff=False)
        jaynes.run(thunk)

    jaynes.listen()
