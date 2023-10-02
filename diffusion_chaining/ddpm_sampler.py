from functools import partial

import numpy as np
import torch
from params_proto import PrefixProto, Proto
from tqdm import tqdm

from diffusion_chaining.ddpm import marginal_prob_std, diffusion_coeff


class DDPM(PrefixProto, cli=False):
    ckpt = None
    model_path = "checkpoints/model_last.pt"

    sigma = 25.0
    snr = Proto(0.16, help="Signal-to-noise ratio.")
    n_steps = Proto(500, help="The number of sampling steps.")
    sample_batch_size = 8 * 8

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"


def pc_sampler(
    score_model,
    marginal_prob_std: callable,
    diffusion_coeff: callable,
    batch_size=DDPM.sample_batch_size,
    time_steps=DDPM.n_steps,
    snr: float = DDPM.snr,
    device="cuda",
    eps=1e-3,
    history=False,
):
    """Generate samples from score-based models with Predictor-Corrector method.

    Args:
      score_model: A PyTorch model that represents the time-dependent score-based model.
      marginal_prob_std: A function that gives the standard deviation
        of the perturbation kernel.
      diffusion_coeff: A function that gives the diffusion coefficient
        of the SDE.
      batch_size: The number of samplers to generate by calling this function once.
      time_steps: The number of sampling steps.
        Equivalent to the number of discretized time steps.
      device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
      eps: The smallest time step for numerical stability.

    Returns:
      Samples.
    """
    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, score_model.input_channels, 28, 28, device=device) * marginal_prob_std(t)[:, None, None, None]
    time_steps = np.linspace(1.0, eps, time_steps)
    step_size = time_steps[0] - time_steps[1]
    x = init_x

    if history:
        # silence the progress when history is needed
        step_iter = time_steps
    else:
        step_iter = tqdm(time_steps, desc="Sampling")

    xs = []
    for time_step in step_iter:  # tqdm(time_steps, desc="Sampling"):
        # for time_step in tqdm(time_steps, desc="Sampling"):
        batch_time_step = torch.ones(batch_size, device=device) * time_step
        # Corrector step (Langevin MCMC)
        grad = score_model(x, batch_time_step)
        grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
        noise_norm = np.sqrt(np.prod(x.shape[1:]))
        langevin_step_size = 2 * (snr * noise_norm / grad_norm) ** 2
        x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)

        # Predictor step (Euler-Maruyama)
        g = diffusion_coeff(batch_time_step)
        x_mean = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
        x = x_mean + torch.sqrt(g**2 * step_size)[:, None, None, None] * torch.randn_like(x)

        if history:
            xs += [x_mean.detach()]

    if history:
        # The last step does not include any noise
        return x_mean, torch.stack(xs), time_steps

    return x_mean


def collect_images(model=None, key="figures/samples.png"):
    from ml_logger import logger

    if model is None:
        from os import path

        model = logger.torch_load(DDPM.ckpt, DDPM.model_path, map_location=DDPM.device)

    ## Generate samples using the specified sampler.
    with torch.no_grad():
        samples = pc_sampler(
            model,
            partial(marginal_prob_std, sigma=DDPM.sigma),
            partial(diffusion_coeff, sigma=DDPM.sigma),
            DDPM.sample_batch_size,
            device=DDPM.device,
        )

    ## Sample visualization.
    samples = samples.clamp(0.0, 1.0)

    from torchvision.utils import make_grid

    composite = make_grid(samples, nrow=int(np.sqrt(DDPM.sample_batch_size)))
    composite = composite.permute(1, 2, 0).cpu().numpy()
    logger.save_image(composite, key)


def main():
    from ml_logger import logger

    print(logger.get_dash_url())
    logger.job_started()

    # fmt: off
    logger.log_text("""
    charts:
    - type: image
      glob: "**/*.png"
    """, filename=".charts.yml", dedent=True, overwrite=True)
    # fmt: on

    collect_images()

    logger.job_completed()

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(6, 6))
    # plt.axis("off")
    # plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0.0, vmax=1.0)
    # plt.show()


if __name__ == "__main__":
    DDPM.ckpt = "/toy-diffusion/toy-diffusion/neurips/ddpm/m1/100"
    DDPM.model_path = "m1_last.pt"
    main()
