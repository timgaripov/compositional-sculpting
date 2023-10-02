import copy
import functools

import numpy as np
import torch
import torch.optim as optim
from ml_logger.job import RUN
from params_proto import PrefixProto
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from diffusion_chaining.bcomp_sampler import composite_factory
from diffusion_chaining.ddpm import marginal_prob_std, diffusion_coeff
from diffusion_chaining.ddpm_sampler import pc_sampler
from diffusion_chaining.models.classifier_model import Classifier2ord
from diffusion_chaining.models.util import set_seed


# Classifier training
@torch.no_grad()
def test(model, score_model1, score_model2, batch_size, device):
    from ml_logger import logger

    target_model = model

    mps_fn = functools.partial(marginal_prob_std, sigma=DDPM_chain.sigma)
    dc_fn = functools.partial(diffusion_coeff, sigma=DDPM_chain.sigma)

    with torch.no_grad():
        x_1, batch1, time_steps1 = pc_sampler(score_model1, mps_fn, dc_fn, batch_size=batch_size, device=device, history=True)
        x_2, batch2, time_steps2 = pc_sampler(score_model2, mps_fn, dc_fn, batch_size=batch_size, device=device, history=True)

    # compute terminal loss
    x_term = torch.cat([x_1, x_2], dim=0)
    time_term = torch.cat(
        [torch.ones(x_1.shape[0], device=device) * time_steps1[-1], torch.ones(x_2.shape[0], device=device) * time_steps2[-1]], dim=0
    )
    logprobs_term = model(x_term, time_term)
    log_p_y_eq_1 = torch.logsumexp(logprobs_term, dim=1)[:, 0]
    log_p_y_eq_2 = torch.logsumexp(logprobs_term, dim=1)[:, 1]

    ce_target_term = torch.cat([torch.zeros(x_1.shape[0], device=device), torch.ones(x_2.shape[0], device=device)], dim=0)
    loss_t0 = -torch.mean(ce_target_term * log_p_y_eq_2 + (1.0 - ce_target_term) * log_p_y_eq_1)
    print(f"Average terminal loss: {loss_t0.item():5f}")
    logger.store_metrics({f"eval/clrf_t0": loss_t0.item(), "eval/t": 0})

    # non-terminal states
    logprobs_term_ema = target_model(x_term, time_term)
    p_x_y2_eq_1 = torch.sum(logprobs_term_ema.exp(), dim=1)[:, 0]
    p_x_y2_eq_2 = torch.sum(logprobs_term_ema.exp(), dim=1)[:, 1]

    for stepIDX in range(0, batch1.shape[0] - 1, 50):
        s_1 = batch1[stepIDX, ...]
        s_2 = batch2[stepIDX, ...]
        s_non_term = torch.cat([s_1, s_2], dim=0)

        time_term = torch.cat(
            [
                torch.ones(s_1.shape[0], device=device) * time_steps1[stepIDX],
                torch.ones(s_2.shape[0], device=device) * time_steps2[stepIDX],
            ],
            dim=0,
        )
        logprobs_non_term = model(s_non_term, time_term)

        w_mat = torch.zeros((s_non_term.shape[0], 2, 2), device=device)
        # set y1 = 0
        w_mat[: s_1.shape[0], 0, 0] = 1.0
        w_mat[: s_1.shape[0], 0, 1] = 1.0
        # set y2 = 1
        w_mat[s_1.shape[0] :, 1, 0] = 1.0
        w_mat[s_1.shape[0] :, 1, 1] = 1.0

        w_mat[:, :, 0] *= p_x_y2_eq_1[:, None]
        w_mat[:, :, 1] *= p_x_y2_eq_2[:, None]

        loss_t = -torch.mean(w_mat * logprobs_non_term)
        print(f"Average Loss at step {time_steps1[stepIDX]:2f}: {loss_t.item():5f}")
        logger.store_metrics({f"eval/clrf_t{time_steps1[stepIDX]}": loss_t.item()})


def loss_first_order():
    pass


def loss_second_order():
    pass


def train(model, target_model, optimizer, score_model1, score_model2, batch_size, device, progress_bar, warmed_up=False):
    from ml_logger import logger

    # target_model = model

    mps_fn = functools.partial(marginal_prob_std, sigma=DDPM_chain.sigma)
    dc_fn = functools.partial(diffusion_coeff, sigma=DDPM_chain.sigma)

    # needed to avoid OOM
    with torch.no_grad():
        x_1, batch1, time_steps1 = pc_sampler(score_model1, mps_fn, dc_fn, batch_size=batch_size, device=device, history=True)
        x_2, batch2, time_steps2 = pc_sampler(score_model2, mps_fn, dc_fn, batch_size=batch_size, device=device, history=True)

    # compute terminal loss
    x_term = torch.cat([x_1, x_2], dim=0)
    time_term = torch.cat(
        [
            torch.ones(x_1.shape[0], device=device) * time_steps1[-1],
            torch.ones(x_2.shape[0], device=device) * time_steps2[-1],
        ],
        dim=0,
    )

    logprobs_term = model(x_term, time_term)

    log_p_y_eq_1 = torch.logsumexp(logprobs_term, dim=1)[:, 0]
    log_p_y_eq_2 = torch.logsumexp(logprobs_term, dim=1)[:, 1]

    ce_target_term = torch.cat(
        [
            torch.zeros(x_1.shape[0], device=device),
            torch.ones(x_2.shape[0], device=device),
        ],
        dim=0,
    )
    loss_t0 = -torch.mean(ce_target_term * log_p_y_eq_2 + (1.0 - ce_target_term) * log_p_y_eq_1)
    logger.store_metrics(**{"loss/clfr_t0": loss_t0.item()})
    loss = loss_t0

    # non-terminal states
    if warmed_up:
        loss_non_term = 0
        with torch.no_grad():
            logprobs_term_ema = target_model(x_term, time_term)
            p_x_y2_eq_1 = torch.sum(logprobs_term_ema.exp(), dim=1)[:, 0]
            p_x_y2_eq_2 = torch.sum(logprobs_term_ema.exp(), dim=1)[:, 1]

        loss_step_weight = 1.0 / (batch1.shape[0] * (1 - DDPM_chain.exclude_from_t) * DDPM_chain.train_fraction)
        for stepIDX in range(batch1.shape[0] - 1):
            if time_steps1[stepIDX] > DDPM_chain.exclude_from_t:
                continue
            if np.random.rand() > DDPM_chain.train_fraction:
                continue

            s_1 = batch1[stepIDX, ...]
            s_2 = batch2[stepIDX, ...]
            s_non_term = torch.cat([s_1, s_2], dim=0)

            time_term = torch.cat(
                [
                    torch.ones(s_1.shape[0], device=device) * time_steps1[stepIDX],
                    torch.ones(s_2.shape[0], device=device) * time_steps2[stepIDX],
                ],
                dim=0,
            )
            logprobs_non_term = model(s_non_term, time_term)

            w_mat = torch.zeros((s_non_term.shape[0], 2, 2), device=device)
            # set y1 = 0
            w_mat[: s_1.shape[0], 0, 0] = 1.0
            w_mat[: s_1.shape[0], 0, 1] = 1.0
            # set y2 = 1
            w_mat[s_1.shape[0] :, 1, 0] = 1.0
            w_mat[s_1.shape[0] :, 1, 1] = 1.0

            w_mat[:, :, 0] *= p_x_y2_eq_1[:, None]
            w_mat[:, :, 1] *= p_x_y2_eq_2[:, None]

            loss_non_term -= torch.mean(w_mat * logprobs_non_term) * loss_step_weight

        loss += loss_non_term
        logger.store_metrics({"loss/clfr_t": loss_non_term.item()})

    progress_bar.set_description("Average Loss: {:5f}".format(loss.item()))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


class DDPM_chain(PrefixProto, cli=False):
    dist_1 = None
    dist_2 = None

    gen_1 = None
    # gen_2 path is not used, use dist_2 directly to load from composite_factory
    # gen_2 = None
    model_path = "checkpoints/model_last.pt"

    seed = 100
    n_epochs = 200

    sigma = 25.0
    batch_size = 64
    exclude_from_t = 0.7  # do not train from this timestep until t = 1.0. This is because the last timesteps are too noisy to train on.
    train_fraction = 0.1  # train on a fraction of randomly selected steps of this size
    cp_interval = 50
    eval_interval = 50

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"


def main(**deps):
    from ml_logger import logger

    print(logger.get_dash_url())

    DDPM_chain._update(deps)
    logger.log_params(DDPM_comp=vars(DDPM_chain))
    # fmt: off
    logger.log_text("""
    charts:
    - yKeys: [loss/clfr_t0/mean, loss/clfr_t/mean]
      xKey: epoch
    """, ".charts.yml", True, True)

    set_seed(DDPM_chain.seed)

    # fmt: on
    gen_1 = logger.torch_load(DDPM_chain.gen_1, DDPM_chain.model_path, map_location=DDPM_chain.device)
    gen_1.requires_grad_(False)

    gen_2 = composite_factory(DDPM_chain.dist_2, 20.0, DDPM_chain.device)

    model = Classifier2ord(input_channels=gen_1.input_channels).to(DDPM_chain.device)
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)

    scheduler = StepLR(optimizer, step_size=1, gamma=0.97)
    for epoch in (bar := tqdm(range(1, DDPM_chain.n_epochs + 1), leave=True, desc="Training")):

        target_model = copy.deepcopy(model)
        target_model.requires_grad_(False)

        warmed_up = epoch >= 100

        if warmed_up and epoch % DDPM_chain.eval_interval == 0:
            test(model, gen_1, gen_2, 256, DDPM_chain.device)

        if epoch % DDPM_chain.cp_interval == 0:
            logger.torch_save(model, f"checkpoints/model_{epoch:04d}.pt")
            logger.duplicate(f"checkpoints/model_{epoch:04d}.pt", f"checkpoints/model_last.pt")

        train(model, target_model, optimizer, gen_1, gen_2, DDPM_chain.batch_size, DDPM_chain.device, bar, warmed_up)

        logger.log_metrics_summary(key_values={"epoch": epoch})

        scheduler.step()


if RUN.debug and __name__ == "__main__":
    from ml_logger.job import instr

    thunk = instr(main)

    thunk(
        **{
            "DDPM_chain.dist_1": "m_a",
            "DDPM_chain.gen_1": "/toy-diffusion/toy-diffusion/neurips/ddpm/base/m_a/100",
            "DDPM_chain.dist_2": "m_bxm_c",
        }
    )

if __name__ == "__main__":
    import jaynes

    from ml_logger.job import instr
    from params_proto.hyper import Sweep
    from ml_logger import logger

    # jaynes.config("local")
    sweep = Sweep(DDPM_chain, RUN).load("analysis/sweeps/chain.jsonl")

    gpus_to_use = [0, 1, 2, 3]

    for i, deps in enumerate(sweep):
        RUN.CUDA_VISIBLE_DEVICES = str(gpus_to_use[i % len(gpus_to_use)])
        jaynes.config("local")
        thunk = instr(main, **deps, __diff=False)
        jaynes.run(thunk)

    jaynes.listen()
