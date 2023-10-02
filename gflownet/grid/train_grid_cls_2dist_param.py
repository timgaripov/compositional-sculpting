import copy
import math
import random
import numpy as np
import torch

import matplotlib.pyplot as plt

from params_proto import ParamsProto


class Args(ParamsProto, prefix='classifier-2dist'):
    device = torch.device('cpu')
    seed = 100

    run_path_1 = None
    run_path_2 = None

    logit_alpha_range = [-3.5, 3.5]

    horizon = 32
    ndim = 2

    num_hidden = 256
    num_layers = 2

    batch_size = 64

    num_training_steps = 15_000

    learning_rate = 0.001
    target_network_ema = 0.995
    loss_non_term_weight_steps = 3_000

    log_every = 250
    eval_every = 1000
    save_every = 1000


from gflownet.grid.train_grid import Args as BaseArgs
from gflownet.grid.train_grid import compute_exact_logp, make_model, get_fwd_logits_fn


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


def toin(z, horizon):
    # [batch_size, ndim] -> [batch_size, ndim * horizon]
    return torch.nn.functional.one_hot(z, horizon).view(z.shape[0], -1).float()


class JointYClassifierParam(torch.nn.Module):
    def __init__(self, horizon, ndim, num_hidden, num_layers):
        super().__init__()
        self.trunk = make_mlp([ndim * horizon + 1] + [num_hidden] * num_layers)
        self.non_term_head = torch.nn.Linear(num_hidden + 1, 3)
        self.term_head = torch.nn.Linear(num_hidden, 1)

    def get_outputs(self, x, logit_alpha, terminal):
        # x: [batch_size, ndim * horizon]
        # logit_alpha [batch_size]
        # terminal: [batch_size] 0.0 or 1.0
        cond = logit_alpha * (1.0 - terminal)
        x = self.trunk(torch.cat((x, cond[:, None]), dim=1))

        non_term_outputs = self.non_term_head(torch.cat((x, cond[:, None]), dim=1))
        term_outputs = self.term_head(x)

        return non_term_outputs, term_outputs

    def forward(self, x, logit_alpha, terminal):
        # x: [batch_size, ndim * horizon]
        # logit_alpha [batch_size]
        # terminal: [batch_size] 0.0 or 1.0

        non_term_outputs, term_outputs = self.get_outputs(x, logit_alpha, terminal)

        # log_probs shape [batch_size, 2x2]
        # non-term probs:
        non_term_tmp = torch.cat([non_term_outputs, torch.zeros_like(non_term_outputs[:, :1])], dim=1)
        non_term_log_probs = torch.log_softmax(non_term_tmp, dim=1)

        # term probs:
        # p(y_1 = 1) = a
        # p(y_1 = 2) = b

        # p(y_2 = 1) = c
        # p(y_2 = 2) = d

        # p(y_1 = 1, y_2 = 1) = ac
        # p(y_1 = 2, y_2 = 2) = bd
        # p(y_1 = 1, y_2 = 2) = ad
        # p(y_1 = 2, y_2 = 1) = bc

        # log p(y_1 = 1, y_2 = 1) = log a + log c
        # log p(y_1 = 2, y_2 = 2) = log b + log d
        # log p(y_1 = 1, y_2 = 2) = log a + log d
        # log p(y_1 = 2, y_2 = 1) = log b + log c

        term_log_a = torch.nn.functional.logsigmoid(-term_outputs)
        term_log_b = torch.nn.functional.logsigmoid(term_outputs)
        term_log_c = torch.nn.functional.logsigmoid(-(term_outputs - logit_alpha[:, None]))
        term_log_d = torch.nn.functional.logsigmoid(term_outputs - logit_alpha[:, None])

        term_log_ab = torch.cat([term_log_a, term_log_b], dim=1)
        term_log_cd = torch.cat([term_log_c, term_log_d], dim=1)

        term_log_probs = (term_log_ab[:, :, None] + term_log_cd[:, None, :]).view(-1, 4)

        log_probs = non_term_log_probs * (1.0 - terminal.view(-1, 1)) + term_log_probs * terminal.view(-1, 1)
        log_probs = log_probs.view(-1, 2, 2)

        return log_probs


def load_model(run_path, device):
    from ml_logger import ML_Logger
    loader = ML_Logger(prefix=run_path)
    BaseArgs._update(**loader.read_params('Args'))

    model, _ = make_model(BaseArgs.horizon, BaseArgs.ndim, BaseArgs.num_hidden, BaseArgs.num_layers)
    model.to(device)
    saved_state = loader.torch_load('checkpoints/model_last.pt', map_location=device)
    model.load_state_dict(saved_state['model'])

    return model


@torch.no_grad()
def sample_from_model(model, horizon, ndim, num_samples, device, return_trajectories=False):
    # [batch_size, ndim]
    z = torch.zeros((num_samples, ndim), device=device, dtype=torch.long)
    trajectories = None
    if return_trajectories:
        trajectories = [z[i].clone()[None, :] for i in range(num_samples)]

    done = torch.full((num_samples,), False, dtype=torch.bool).to(device)

    while torch.any(~done):
        pred = model(toin(z[~done], horizon))

        edge_mask = torch.cat([(z[~done] == horizon - 1).float(),
                               torch.zeros(((~done).sum(), 1), device=device)], 1)
        logits = (pred[..., :ndim + 1] - INF * edge_mask).log_softmax(1)

        sample_ins_probs = logits.softmax(1)
        sample_ins_probs = sample_ins_probs

        action = sample_ins_probs.multinomial(1)

        terminate = (action == ndim).squeeze(1)

        done[~done] |= terminate

        # update state
        z[~done] = z[~done].scatter_add(1, action[~terminate],
                                        torch.ones(action[~terminate].shape,
                                                   dtype=torch.long, device=device))

        for i in torch.nonzero(~done).squeeze(1):
            if return_trajectories:
                trajectories[i] = torch.cat([trajectories[i], z[i].clone()[None, :]], dim=0)

    if return_trajectories:
        return z, trajectories
    return z


def get_joint_guided_fwd_logits_fn(model_1, model_2, cls_main, horizon, ndim, device,
                                   just_mixture=False, y1=1, y2=2, logit_alpha=0.0):
    if y1 not in {1, 2} or y2 not in {1, 2}:
        raise ValueError(f'Invalid y1 or y2: {y1}, {y2}')

    def guided_fwd_logits_fn(z):
        # z: [batch_size, ndim]
        enc = toin(z, horizon)

        model_fwd_logits_1 = model_1(enc.to(device))[:, :ndim + 1]
        model_fwd_logits_2 = model_2(enc.to(device))[:, :ndim + 1]
        model_fwd_logprobs_1 = model_fwd_logits_1.log_softmax(dim=1)
        model_fwd_logprobs_2 = model_fwd_logits_2.log_softmax(dim=1)

        z_next = z[:, None, :] + torch.eye(ndim, dtype=torch.long)[None, :, :]
        z_next_valid_mask = torch.all(z_next < horizon, dim=2)
        # clip at horizion - 1 to make one_hot work
        z_next = torch.minimum(z_next, torch.tensor(horizon - 1, device=z_next.device))
        z_next = z_next.view(-1, ndim)

        logit_alpha_tensor = torch.full((z.shape[0],), logit_alpha, device=device)

        cls_logprobs_cur = cls_main(toin(z, horizon).to(device),
                                    logit_alpha_tensor,
                                    torch.zeros(z.shape[0], device=device))

        logp_y1_eq_1_cur = torch.logsumexp(cls_logprobs_cur, dim=2)[:, 0]
        logp_y1_eq_2_cur = torch.logsumexp(cls_logprobs_cur, dim=2)[:, 1]

        mixture_logits = torch.logsumexp(
            torch.stack([model_fwd_logprobs_1 + logp_y1_eq_1_cur[:, None],
                         model_fwd_logprobs_2 + logp_y1_eq_2_cur[:, None]], dim=0),
            dim=0)
        if just_mixture:
            return mixture_logits

        logit_alpha_tensor_next = torch.full((z_next.shape[0],), logit_alpha, device=device)

        cls_logprobs_next = cls_main(toin(z_next, horizon).to(device),
                                     logit_alpha_tensor_next,
                                     torch.zeros(z_next.shape[0], device=device))
        cls_logprobs_next = cls_logprobs_next.view(z.shape[0], ndim, 2, 2)

        cls_logprobs_end = cls_main(toin(z, horizon).to(device),
                                    logit_alpha_tensor,
                                    torch.ones(z.shape[0], device=device))

        guidance_next = cls_logprobs_next[:, :, y1 - 1, y2 - 1] - cls_logprobs_cur[:, None, y1 - 1, y2 - 1]
        guidance_next[~z_next_valid_mask] = 0.0

        guidance_end = cls_logprobs_end[:, y1 - 1, y2 - 1] - cls_logprobs_cur[:, y1 - 1, y2 - 1]

        guidance = torch.cat([guidance_next, guidance_end[:, None]], dim=1)

        return mixture_logits + guidance

    return guided_fwd_logits_fn


def main(**deps):
    Args._update(deps)
    set_seed(Args.seed)

    from ml_logger import logger

    logger.log_params(Args=vars(Args))
    logger.log_text("""
    charts:
    - type: image
      glob: dist_figs/p1.png
    - type: image
      glob: dist_figs/p2.png
    - type: image
      glob: dist_figs/gt_hm_005.png
    - type: image
      glob: dist_figs/gt_hm_050.png
    - type: image
      glob: dist_figs/gt_hm_095.png
    - type: image
      glob: dist_figs/gt_diff_005.png
    - type: image
      glob: dist_figs/gt_diff_050.png
    - type: image
      glob: dist_figs/gt_diff_095.png
    - type: image
      glob: dist_figs/gt_mixture.png
    - type: image
      glob: dist_figs/hm_005_step_*.png
    - type: image
      glob: dist_figs/hm_050_step_*.png
    - type: image
      glob: dist_figs/hm_095_step_*.png
    - type: image
      glob: dist_figs/diff_005_step_*.png
    - type: image
      glob: dist_figs/diff_050_step_*.png
    - type: image
      glob: dist_figs/diff_095_step_*.png
    - type: image
      glob: dist_figs/mixture_step_*.png
    - yKey: loss/mean
      xKey: step
    - yKey: loss_term/mean
      xKey: step
    - yKey: loss_non_term/mean
      xKey: step
    - yKey: loss_non_term_weight/mean
      xKey: step
    - yKey: grad_norm/mean
      xKey: step
    - yKey: param_norm/mean
      xKey: step
    - yKeys: ["hm_005_dist_l1/mean", "hm_050_dist_l1/mean", "hm_095_dist_l1/mean"]
      xKey: step
    - yKeys: ["diff_005_dist_l1/mean","diff_050_dist_l1/mean","diff_095_dist_l1/mean"]
      xKey: step
    - yKey: mixture_dist_l1/mean
      xKey: step
    - yKey: steps_per_sec/mean
      xKey: step
    """, ".charts.yml", dedent=True)

    model_1 = load_model(Args.run_path_1, Args.device)
    model_2 = load_model(Args.run_path_2, Args.device)

    fwd_logits_fn_1 = get_fwd_logits_fn(model_1, Args.horizon, Args.ndim, Args.device)
    logp_1 = compute_exact_logp(fwd_logits_fn_1, Args.horizon, Args.ndim, Args.device)
    fwd_logits_fn_2 = get_fwd_logits_fn(model_2, Args.horizon, Args.ndim, Args.device)
    logp_2 = compute_exact_logp(fwd_logits_fn_2, Args.horizon, Args.ndim, Args.device)

    alpha_strs = ["005", "050", "095"]

    logp_hm_gt_list = []
    disrt_hm_gt_list = []

    logp_diff_gt_list = []
    disrt_diff_gt_list = []

    for alpha_str in alpha_strs:
        alpha = float(alpha_str) / 100

        logp_hm_gt = logp_1 + logp_2 - \
                     torch.logsumexp(
                         torch.stack([logp_1 + math.log(alpha), logp_2 + math.log(1 - alpha)], dim=0),
                     dim=0)
        logp_hm_gt = torch.log_softmax(logp_hm_gt, dim=0)
        distr_hm_gt = torch.exp(logp_hm_gt)

        logp_hm_gt_list.append(logp_hm_gt)
        disrt_hm_gt_list.append(distr_hm_gt)


        logp_diff_gt = logp_1 + logp_1 - \
                       torch.logsumexp(
                           torch.stack([logp_1 + math.log(alpha), logp_2 + math.log(1 - alpha)], dim=0),
                       dim=0)
        logp_diff_gt = torch.log_softmax(logp_diff_gt, dim=0)
        distr_diff_gt = torch.exp(logp_diff_gt)

        logp_diff_gt_list.append(logp_diff_gt)
        disrt_diff_gt_list.append(distr_diff_gt)


    logp_mixture_gt = torch.logsumexp(torch.stack([logp_1, logp_2], dim=0), dim=0) - np.log(2)
    logp_mixture_gt = torch.log_softmax(logp_mixture_gt, dim=0)
    distr_mixture_gt = torch.exp(logp_mixture_gt)

    def plot_distr(path, distribution, title):
        distribution_2d = distribution.reshape(Args.horizon, Args.horizon).T

        fig = plt.figure(figsize=(10, 10))
        plt.imshow(distribution_2d, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title(title, fontsize=24)

        logger.savefig(path)

    plot_distr('dist_figs/p1.png', logp_1.exp().detach().cpu().numpy(), 'P1')
    plot_distr('dist_figs/p2.png', logp_2.exp().detach().cpu().numpy(), 'P2')

    for alpha_str, distr_hm_gt, distr_diff_gt in zip(alpha_strs, disrt_hm_gt_list, disrt_diff_gt_list):
        plot_distr(f'dist_figs/gt_hm_{alpha_str}.png', distr_hm_gt.detach().cpu().numpy(),
                   f'GT Harmonic Mean\nalpha=0.{alpha_str}')
        plot_distr(f'dist_figs/gt_diff_{alpha_str}.png', distr_diff_gt.detach().cpu().numpy(),
                   f'GT diff(P1, P2)\nalpha=0.{alpha_str}')

    plot_distr('dist_figs/gt_mixture.png', logp_mixture_gt.exp().detach().cpu().numpy(), 'Ground Truth\nMixture')

    def save(cls, target_cls, opt, suffix='_last'):
        logger.torch_save({
            'cls': cls.state_dict(),
            'target_cls': target_cls.state_dict(),
            'opt': opt.state_dict(),
        }, f'checkpoints/model{suffix}.pt')


    cls = JointYClassifierParam(Args.horizon, Args.ndim, Args.num_hidden, Args.num_layers)

    target_cls = copy.deepcopy(cls)
    for p in target_cls.parameters():
        p.requires_grad = False

    cls.to(Args.device)
    target_cls.to(Args.device)

    opt = torch.optim.Adam(cls.parameters(), lr=Args.learning_rate)

    logger.start('log_timer')
    timer_steps = 0

    for step in range(Args.num_training_steps):
        x_1, trajectories_1 = sample_from_model(model_1, Args.horizon, Args.ndim, Args.batch_size, Args.device,
                                                return_trajectories=True)
        x_2, trajectories_2 = sample_from_model(model_2, Args.horizon, Args.ndim, Args.batch_size, Args.device,
                                                return_trajectories=True)


        u = torch.rand(2 * Args.batch_size, device=Args.device)
        logit_alpha = Args.logit_alpha_range[0] + (Args.logit_alpha_range[1] - Args.logit_alpha_range[0]) * u

        # compute terminal loss
        x_term = torch.cat([x_1, x_2], dim=0)
        ce_target_term = torch.cat([torch.zeros(x_1.shape[0], device=Args.device),
                                    torch.ones(x_2.shape[0], device=Args.device)], dim=0)

        enc_term = toin(x_term, Args.horizon).to(Args.device)
        logprobs_term = cls(enc_term, logit_alpha, torch.ones(enc_term.shape[0], device=Args.device))
        log_p_y_eq_1 = torch.logsumexp(logprobs_term, dim=2)[:, 0]
        log_p_y_eq_2 = torch.logsumexp(logprobs_term, dim=2)[:, 1]

        # loss_term = torch.nn.functional.binary_cross_entropy_with_logits(log_p_y_eq_2, ce_target_term)
        loss_term = -torch.mean(ce_target_term * log_p_y_eq_2 + (1.0 - ce_target_term) * log_p_y_eq_1)

        # compute non-terminal loss

        s_1 = torch.cat(trajectories_1, dim=0)
        s_2 = torch.cat(trajectories_2, dim=0)
        s_non_term = torch.cat([s_1, s_2], dim=0)
        enc_non_term = toin(s_non_term, Args.horizon).to(Args.device)
        traj_lens = [traj.shape[0] for traj in trajectories_1 + trajectories_2]
        traj_lens = torch.tensor(traj_lens, device=Args.device)
        traj_ind = torch.arange(0, traj_lens.shape[0], device=Args.device)
        traj_ind = traj_ind.repeat_interleave(traj_lens)

        with torch.no_grad():
            _, term_outputs_ema = target_cls.get_outputs(enc_term,
                                                         logit_alpha,
                                                         torch.ones(enc_term.shape[0], device=Args.device))

            # use alpha to compute p(y2|x)
            p_x_y2_eq_1 = torch.sigmoid(-(term_outputs_ema - logit_alpha[:, None])).squeeze()
            p_x_y2_eq_2 = torch.sigmoid(term_outputs_ema - logit_alpha[:, None]).squeeze()

        logprobs_non_term = cls(enc_non_term, logit_alpha[traj_ind],
                                torch.zeros(enc_non_term.shape[0], device=Args.device))

        w_s_y2_eq_1 = p_x_y2_eq_1[traj_ind]
        w_s_y2_eq_2 = p_x_y2_eq_2[traj_ind]

        w_mat = torch.zeros((s_non_term.shape[0], 2, 2), device=Args.device)
        # set y1 = 0
        w_mat[:s_1.shape[0], 0, 0] = 1.0
        w_mat[:s_1.shape[0], 0, 1] = 1.0
        # set y1 = 1
        w_mat[s_1.shape[0]:, 1, 0] = 1.0
        w_mat[s_1.shape[0]:, 1, 1] = 1.0

        w_mat[:, :, 0] *= w_s_y2_eq_1[:, None]
        w_mat[:, :, 1] *= w_s_y2_eq_2[:, None]

        loss_non_term = -torch.sum(w_mat * logprobs_non_term) / (2 * Args.batch_size)

        loss_non_term_weight = 1.0
        if Args.loss_non_term_weight_steps > 0:
            loss_non_term_weight = min(1.0, step / Args.loss_non_term_weight_steps)

        loss = loss_term + loss_non_term * loss_non_term_weight

        opt.zero_grad()
        loss.backward()
        opt.step()

        # update target network
        for a, b in zip(cls.parameters(), target_cls.parameters()):
            b.data.mul_(Args.target_network_ema).add_(a.data * (1 - Args.target_network_ema))

        timer_steps += 1
        grad_norm = sum([p.grad.detach().norm() ** 2 for p in cls.parameters()]) ** 0.5
        param_norm = sum([p.detach().norm() ** 2 for p in cls.parameters()]) ** 0.5
        logger.store_metrics({
            'loss': loss.item(),
            'grad_norm': grad_norm.item(),
            'param_norm': param_norm.item(),
            'loss_term': loss_term.item(),
            'loss_non_term': loss_non_term.item(),
            'loss_non_term_weight': loss_non_term_weight,
        })

        if step % Args.save_every == 0:
            save(cls, target_cls, opt)

        if step % Args.eval_every == 0:
            for alpha_str, distr_hm_gt, distr_diff_gt in zip(alpha_strs, disrt_hm_gt_list, disrt_diff_gt_list):
                alpha = float(alpha_str) / 100.0
                logit_alpha = math.log(alpha) - math.log(1.0 - alpha)
                # HERE

                fwd_logits_hm_fn = get_joint_guided_fwd_logits_fn(model_1, model_2, cls, Args.horizon, Args.ndim,
                                                                  Args.device, y1=1, y2=2, logit_alpha=logit_alpha)
                logp_hm_model = compute_exact_logp(fwd_logits_hm_fn, Args.horizon, Args.ndim, Args.device)
                distr_hm_model = logp_hm_model.exp()
                assert (distr_hm_model.sum() - 1.0).abs() < 1e-4

                l1_key = f'hm_{alpha_str}_dist_l1'
                logger.store_metrics(**{l1_key: np.abs(distr_hm_model - distr_hm_gt).sum()})
                plot_distr(f'dist_figs/hm_{alpha_str}_step_{step:08d}.png',
                           distr_hm_model, f'Generated distribution at step {step}')

                fwd_logits_diff_fn = get_joint_guided_fwd_logits_fn(model_1, model_2, cls, Args.horizon, Args.ndim,
                                                                    Args.device, y1=1, y2=1, logit_alpha=logit_alpha)
                logp_diff_model = compute_exact_logp(fwd_logits_diff_fn, Args.horizon, Args.ndim, Args.device)
                distr_diff_model = logp_diff_model.exp()
                assert (distr_diff_model.sum() - 1.0).abs() < 1e-4

                l1_key = f'diff_{alpha_str}_dist_l1'
                logger.store_metrics(**{l1_key: np.abs(distr_diff_model - distr_diff_gt).sum()})
                plot_distr(f'dist_figs/diff_{alpha_str}_step_{step:08d}.png',
                           distr_diff_model, f'Generated distribution at step {step}')


            fwd_logits_mixture_fn = get_joint_guided_fwd_logits_fn(model_1, model_2, cls, Args.horizon, Args.ndim,
                                                                   Args.device, just_mixture=True)
            logp_mixture_model = compute_exact_logp(fwd_logits_mixture_fn, Args.horizon, Args.ndim, Args.device)
            distr_mixture_model = logp_mixture_model.exp()
            assert (distr_mixture_model.sum() - 1.0).abs() < 1e-4

            logger.store_metrics(mixture_dist_l1=np.abs(distr_mixture_model - distr_mixture_gt).sum())
            plot_distr(f'dist_figs/mixture_step_{step:08d}.png', distr_mixture_model, f'Generated distribution at step {step}')

        if step % Args.log_every == 0:
            logger.store_metrics({
                'steps_per_sec': timer_steps / logger.split('log_timer')
            })
            timer_steps = 0
            logger.log_metrics_summary(key_values={'step': step})


if __name__ == '__main__':
    from ml_logger import instr
    thunk = instr(main)
    thunk()
