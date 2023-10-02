# Adapted from https://gist.githubusercontent.com/malkin1729/9a87ce4f19acdc2c24225782a8b81c15/raw/72b2f2272a5bb5b8da460b183816d7b9ba4e5f76/grid.py

import pathlib
import random
import time
import numpy as np
import torch

import matplotlib.pyplot as plt
import seaborn as sns

from params_proto import ParamsProto

from gflownet.grid.rewards import LogRewardFns


class Args(ParamsProto, prefix='gflownet'):

    device = torch.device('cpu')
    seed = 100

    horizon = 32
    ndim = 2

    reward_name = 'diag_sigmoid'
    reward_temperature = 1.0

    num_hidden = 256
    num_layers = 2

    batch_size = 16

    num_training_steps = 20_000

    learning_rate = 0.001
    log_Z_learning_rate = 0.1

    uniform_pb = True
    random_action_prob = 0.05

    log_every = 250
    eval_every = 1000
    save_every = 1000


INF = 1e9


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_mlp(l, act=torch.nn.LeakyReLU(), tail=()):
    return torch.nn.Sequential(*(sum(
        [[torch.nn.Linear(i, o)] + ([act] if n < len(l) - 2 else [])
         for n, (i, o) in enumerate(zip(l, l[1:]))], []) + list(tail)))


def make_model(horizon, ndim, num_hidden, num_layers):
    model = make_mlp([ndim * horizon] + [num_hidden] * num_layers + [2 * ndim + 1])
    log_Z = torch.zeros((1,))
    return model, log_Z


def toin(z, horizon):
    # [batch_size, ndim] -> [batch_size, ndim * horizon]
    return torch.nn.functional.one_hot(z, horizon).view(z.shape[0], -1).float()


def get_fwd_logits_fn(model, horizon, ndim, device):
    def fwd_logits_fn(z):
        x = toin(z, horizon)
        return model(x.to(device))[:, :ndim + 1]
    return fwd_logits_fn


@torch.no_grad()
def compute_exact_logp(fwd_logits_fn, horizon, ndim, device):
    pos = torch.zeros((horizon,) * ndim + (ndim,), dtype=torch.long)
    for i in range(ndim):
        pos_i = torch.arange(0, horizon)
        for _ in range(i):
            pos_i = pos_i.unsqueeze(1)
        pos[..., i] = pos_i

    z_all = pos.view(-1, ndim)
    fwd_logits = fwd_logits_fn(z_all).cpu()

    edge_mask = torch.cat([(z_all == horizon - 1).float(),
                            torch.zeros((z_all.shape[0], 1))], dim=1)
    fwd_logprobs = (fwd_logits - INF * edge_mask).log_softmax(1)

    logp_visit = torch.zeros((horizon ** ndim,))
    logp_end = torch.zeros((horizon ** ndim,))
    num_unprocessed_parents = (z_all != 0).sum(1)

    n_queue = [x.item() for x in horizon ** torch.arange(ndim)]

    # process 0
    for n in n_queue:
        num_unprocessed_parents[n] -= 1

    logp_end[0] = fwd_logprobs[0, -1]

    while n_queue:
        n = n_queue.pop(0)
        assert num_unprocessed_parents[n] == 0

        z = []
        tmp = n
        for _ in range(ndim):
            z.append(tmp % horizon)
            tmp //= horizon

        z = torch.tensor(z)

        z_parents = z[None, :] - torch.eye(ndim, dtype=torch.long)[z > 0]
        a_parents = (z > 0).nonzero().view(-1)
        n_parents = (z_parents * (horizon ** torch.arange(ndim))[None, :]).sum(dim=1)

        log_trans_parents = logp_visit[n_parents] + \
                            torch.gather(fwd_logprobs[n_parents], 1, a_parents[:, None]).view(-1)

        logp_visit[n] = torch.logsumexp(log_trans_parents, dim=0)
        logp_end[n] = logp_visit[n] + fwd_logprobs[n, -1]

        # add children to queue
        z_children = z[None, :] + torch.eye(ndim, dtype=torch.long)[z < horizon - 1]
        n_children = (z_children * (horizon ** torch.arange(ndim))[None, :]).sum(dim=1)
        num_unprocessed_parents[n_children] -= 1
        for n_child in n_children[num_unprocessed_parents[n_children] == 0]:
            n_queue.append(n_child.item())

    return logp_end


def main(**deps):

    Args._update(deps)
    set_seed(Args.seed)

    print(vars(Args))

    from ml_logger import logger
    print(logger)

    logger.log_params(Args=vars(Args))
    logger.log_text("""
    charts:
    - yKey: loss/mean
      xKey: step
    - yKey: log_Z/mean
      xKey: step
    - yKeys: ["log_ratio_min/mean", "log_ratio_mean/mean", "log_ratio_max/mean"]
      xKey: step
    - yKey: grad_norm/mean
      xKey: step
    - yKey: param_norm/mean
      xKey: step
    - yKey: dist_l1/mean
      xKey: step
    - yKey: steps_per_sec/mean
      xKey: step
    - type: image
      glob: dist_figs/step_*.png
    - type: image
      glob: dist_figs/gt.png 
    """, ".charts.yml", dedent=True)

    base_log_reward_fn = getattr(LogRewardFns, Args.reward_name)

    def log_reward(z):
        x_scaled = z / (Args.horizon - 1) * 2 - 1
        base_log_r = base_log_reward_fn(x_scaled)

        return base_log_r / Args.reward_temperature


    pos = torch.zeros((Args.horizon,) * Args.ndim + (Args.ndim,))
    for i in range(Args.ndim):
        pos_i = torch.linspace(0, Args.horizon - 1, Args.horizon)
        for _ in range(i):
            pos_i = pos_i.unsqueeze(1)
        pos[..., i] = pos_i

    truelr = log_reward(pos)
    true_dist = truelr.flatten().softmax(0).cpu().numpy()


    model, log_Z = make_model(Args.horizon, Args.ndim, Args.num_hidden, Args.num_layers)
    log_Z = log_Z.to(Args.device)
    model.to(Args.device)
    opt = torch.optim.Adam([{'params': model.parameters(), 'lr': Args.learning_rate},
                            {'params': [log_Z], 'lr': Args.log_Z_learning_rate}])
    log_Z.requires_grad_()

    losses = []
    log_Zs = []
    all_visited = []
    first_visit = -1 * np.ones_like(true_dist)

    cmap = sns.color_palette("Blues", as_cmap=True)

    def plot_distr(path, distribution, title):
        distribution_2d = distribution.reshape(Args.horizon, Args.horizon).T

        vmax = distribution_2d.max()
        vmin = 0.0 - 0.05 * vmax

        fig = plt.figure(figsize=(10, 10))
        plt.imshow(distribution_2d, cmap=cmap,
                   interpolation='nearest', vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.title(title, fontsize=24)

        logger.savefig(path)

    plot_distr(f'dist_figs/gt.png', true_dist, f'Ground truth')

    def save(model, log_z, opt, suffix='_last'):
        logger.torch_save({
            'model': model.state_dict(),
            'log_z': log_z,
            'opt': opt.state_dict(),
        }, f'checkpoints/model{suffix}.pt')

    logger.start('log_timer')
    timer_steps = 0

    for it in range(Args.num_training_steps):
        opt.zero_grad()

        z = torch.zeros((Args.batch_size, Args.ndim), dtype=torch.long).to(Args.device)
        done = torch.full((Args.batch_size,), False, dtype=torch.bool).to(Args.device)

        action = None

        ll_diff = torch.zeros((Args.batch_size,)).to(Args.device)
        ll_diff += log_Z


        i = 0
        while torch.any(~done):

            pred = model(toin(z[~done], Args.horizon))

            edge_mask = torch.cat([(z[~done] == Args.horizon - 1).float(),
                                   torch.zeros(((~done).sum(), 1), device=Args.device)], 1)
            logits = (pred[..., :Args.ndim + 1] - INF * edge_mask).log_softmax(1)

            init_edge_mask = (z[~done] == 0).float()
            # uniform backward action logtis
            back_logits = torch.zeros_like(init_edge_mask)
            if not Args.uniform_pb:
                back_logits = pred[..., Args.ndim + 1:2 * Args.ndim + 1]
            # apply mask
            back_logits = (back_logits - INF * init_edge_mask).log_softmax(1)

            if action is not None:
                ll_diff[~done] -= back_logits.gather(1, action[action != Args.ndim].unsqueeze(1)).squeeze(1)

            temp = 1.0
            sample_ins_probs = (logits / temp).softmax(1)
            uniform_act_probs = (1.0 - edge_mask) / (1.0 - edge_mask + 0.0000001).sum(1).unsqueeze(1)
            sample_ins_probs = (1.0 - Args.random_action_prob) * sample_ins_probs + Args.random_action_prob * uniform_act_probs

            action = sample_ins_probs.multinomial(1)
            ll_diff[~done] += logits.gather(1, action).squeeze(1)

            terminate = (action == Args.ndim).squeeze(1)
            for x in z[~done][terminate]:
                state = (x.cpu() * (Args.horizon ** torch.arange(Args.ndim))).sum().item()
                if first_visit[state] < 0:
                    first_visit[state] = it
                all_visited.append(state)

            done[~done] |= terminate

            with torch.no_grad():
                # update state
                z[~done] = z[~done].scatter_add(1, action[~terminate],
                                                torch.ones(action[~terminate].shape,
                                                           dtype=torch.long, device=Args.device))

            i += 1

        lr = log_reward(z.float())
        ll_diff -= lr
        loss = (ll_diff ** 2).sum() / Args.batch_size
        loss.backward()
        opt.step()

        grad_norm = sum([p.grad.detach().norm() ** 2 for p in model.parameters()]) ** 0.5
        param_norm = sum([p.detach().norm() ** 2 for p in model.parameters()]) ** 0.5

        losses.append(loss.item())
        log_Zs.append(log_Z.item())

        timer_steps += 1

        logger.store_metrics(loss=loss.item(), log_Z=log_Z.item(),
                             grad_norm=grad_norm.item(), param_norm=param_norm.item(),
                             log_ratio_min=ll_diff.min().item(),
                             log_ratio_mean=ll_diff.mean().item(),
                             log_ratio_max=ll_diff.max().item())

        if it % Args.eval_every == 0:
            fwd_logits_fn = get_fwd_logits_fn(model, Args.horizon, Args.ndim, Args.device)

            logp_model = compute_exact_logp(fwd_logits_fn, Args.horizon, Args.ndim, Args.device)
            distr_model = logp_model.exp()
            assert (distr_model.sum() - 1.0).abs() < 1e-4

            logger.store_metrics(dist_l1=np.abs(distr_model - true_dist).sum())
            plot_distr(f'dist_figs/step_{it:08d}.png', distr_model, f'Generated distribution at step {it}')

        if it % Args.save_every == 0:
            save(model, log_Z, opt)

        if it % Args.log_every == 0:
            logger.store_metrics(steps_per_sec=timer_steps / logger.split('log_timer'))
            timer_steps = 0
            logger.log_metrics_summary(key_values={'step': it})


if __name__ == '__main__':
    from ml_logger import instr
    thunk = instr(main)
    thunk()
