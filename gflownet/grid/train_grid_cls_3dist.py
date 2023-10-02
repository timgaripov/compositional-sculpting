import copy
import math
import random
import numpy as np
import torch

import matplotlib.pyplot as plt
import seaborn as sns

from params_proto import ParamsProto


class Args(ParamsProto, prefix='classifier-3dist'):
    device = torch.device('cpu')
    seed = 100

    run_path_1 = None
    run_path_2 = None
    run_path_3 = None

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


class Joint3YClassifier(torch.nn.Module):
    def __init__(self, horizon, ndim, num_hidden, num_layers):
        super().__init__()
        self.trunk = make_mlp([ndim * horizon] + [num_hidden] * num_layers)
        self.non_term_head = torch.nn.Linear(num_hidden, 9)
        self.term_head = torch.nn.Linear(num_hidden, 2)

    def forward(self, x, terminal):
        # x: [batch_size, ndim * horizon]
        # terminal: [batch_size] 0.0 or 1.0
        x = self.trunk(x)
        non_term_outputs = self.non_term_head(x)
        term_outputs = self.term_head(x)

        # log_probs shape [batch_size, 3x3x3]
        # non-term probs:
        # a + b + c + 3d + 3e + 3f + 3g + 3h + 3i + 6k = 1

        non_term_tmp = torch.cat([non_term_outputs, torch.zeros_like(non_term_outputs[:, :1])], dim=1)
        non_term_tmp = torch.log_softmax(non_term_tmp, dim=1)
        # [batch_size, 10]

        aslice = non_term_tmp[:, :1]
        bslice = non_term_tmp[:, 1:2]
        cslice = non_term_tmp[:, 2:3]
        dslice = non_term_tmp[:, 3:4] - math.log(3.0)
        eslice = non_term_tmp[:, 4:5] - math.log(3.0)
        fslice = non_term_tmp[:, 5:6] - math.log(3.0)
        gslice = non_term_tmp[:, 6:7] - math.log(3.0)
        hslice = non_term_tmp[:, 7:8] - math.log(3.0)
        islice = non_term_tmp[:, 8:9] - math.log(3.0)
        kslice = non_term_tmp[:, 9:10] - math.log(6.0)

        non_term_log_probs = torch.cat([
            aslice,  # 111
            dslice,  # 112
            eslice,  # 113
            dslice,  # 121
            fslice,  # 122,
            kslice,  # 123
            eslice,  # 131
            kslice,  # 132
            hslice,  # 133

            dslice,  # 211
            fslice,  # 212
            kslice,  # 213
            fslice,  # 221
            bslice,  # 222
            gslice,  # 223
            kslice,  # 231
            gslice,  # 232
            islice,  # 233

            eslice,  # 311
            kslice,  # 312
            hslice,  # 313
            kslice,  # 321
            gslice,  # 322
            islice,  # 323
            hslice,  # 331
            islice,  # 332
            cslice,  # 333
        ], dim=1)

        term_logp_single = torch.log_softmax(
            torch.cat([term_outputs, torch.zeros_like(term_outputs[:, :1])], dim=1), dim=1)

        term_log_probs = (
                term_logp_single[:, :, None, None] +
                term_logp_single[:, None, :, None] +
                term_logp_single[:, None, None, :]
        ).view(-1, 27)

        log_probs = non_term_log_probs * (1.0 - terminal.view(-1, 1)) + term_log_probs * terminal.view(-1, 1)
        log_probs = log_probs.view(-1, 3, 3, 3)

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


def get_joint_guided_fwd_logits_fn(model_1, model_2, model_3,
                                   cls_main, horizon, ndim, device,
                                   just_mixture=False, y1=1, y2=2, y3=3):
    if y1 not in {1, 2, 3, None} or y2 not in {1, 2, 3, None} or y3 not in {1, 2, 3, None}:
        raise ValueError(f'Invalid y1 or y2 or y3: {y1}, {y2}, {y3}')

    def guided_fwd_logits_fn(z):
        # z: [batch_size, ndim]
        enc = toin(z, horizon)

        model_fwd_logits_1 = model_1(enc.to(device))[:, :ndim + 1]
        model_fwd_logits_2 = model_2(enc.to(device))[:, :ndim + 1]
        model_fwd_logits_3 = model_3(enc.to(device))[:, :ndim + 1]
        model_fwd_logprobs_1 = model_fwd_logits_1.log_softmax(dim=1)
        model_fwd_logprobs_2 = model_fwd_logits_2.log_softmax(dim=1)
        model_fwd_logprobs_3 = model_fwd_logits_3.log_softmax(dim=1)

        z_next = z[:, None, :] + torch.eye(ndim, dtype=torch.long)[None, :, :]
        z_next_valid_mask = torch.all(z_next < horizon, dim=2)
        # clip at horizion - 1 to make one_hot work
        z_next = torch.minimum(z_next, torch.tensor(horizon - 1, device=z_next.device))
        z_next = z_next.view(-1, ndim)

        cls_logprobs_cur = cls_main(toin(z, horizon).to(device), torch.zeros(z.shape[0], device=device))

        logp_y1_eq_1_cur = torch.logsumexp(cls_logprobs_cur, dim=(2, 3))[:, 0]
        logp_y1_eq_2_cur = torch.logsumexp(cls_logprobs_cur, dim=(2, 3))[:, 1]
        logp_y1_eq_3_cur = torch.logsumexp(cls_logprobs_cur, dim=(2, 3))[:, 2]

        mixture_logits = torch.logsumexp(
            torch.stack([model_fwd_logprobs_1 + logp_y1_eq_1_cur[:, None],
                         model_fwd_logprobs_2 + logp_y1_eq_2_cur[:, None],
                         model_fwd_logprobs_3 + logp_y1_eq_3_cur[:, None]], dim=0),
            dim=0)

        if just_mixture:
            return mixture_logits

        cls_logprobs_next = cls_main(toin(z_next, horizon).to(device), torch.zeros(z_next.shape[0], device=device))
        cls_logprobs_next = cls_logprobs_next.view(z.shape[0], ndim, 3, 3, 3)
        cls_logprobs_end = cls_main(toin(z, horizon).to(device), torch.ones(z.shape[0], device=device))

        def extract_logprobs(logprobs, y1, y2, y3):
            result = logprobs
            if y3 is None:
                result = torch.logsumexp(result, dim=-1)
            else:
                result = result[..., y3 - 1]

            if y2 is None:
                result = torch.logsumexp(result, dim=-1)
            else:
                result = result[..., y2 - 1]

            if y1 is None:
                result = torch.logsumexp(result, dim=-1)
            else:
                result = result[..., y1 - 1]

            return result


        # guidance_next = cls_logprobs_next[:, :, y1 - 1, y2 - 1, y3 - 1] - \
        #                 cls_logprobs_cur[:, None, y1 - 1, y2 - 1, y3 - 1]
        guidance_next = extract_logprobs(cls_logprobs_next, y1, y2, y3) - \
                        extract_logprobs(cls_logprobs_cur[:, None, :, :, :], y1, y2, y3)
        guidance_next[~z_next_valid_mask] = 0.0

        # guidance_end = cls_logprobs_end[:, y1 - 1, y2 - 1, y3 - 1] - \
        #                cls_logprobs_cur[:, y1 - 1, y2 - 1, y3 - 1]
        guidance_end = extract_logprobs(cls_logprobs_end, y1, y2, y3) - \
                       extract_logprobs(cls_logprobs_cur, y1, y2, y3)

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
      glob: dist_figs/p3.png
    - type: image
      glob: dist_figs/gt_12.png
    - type: image
      glob: dist_figs/gt_13.png
    - type: image
      glob: dist_figs/gt_23.png
    - type: image
      glob: dist_figs/gt_123.png
    - type: image
      glob: dist_figs/gt_11.png
    - type: image
      glob: dist_figs/gt_22.png
    - type: image
      glob: dist_figs/gt_33.png
    - type: image
      glob: dist_figs/gt_111.png
    - type: image
      glob: dist_figs/gt_222.png
    - type: image
      glob: dist_figs/gt_333.png
    - type: image
      glob: dist_figs/gt_mixture.png
    - type: image
      glob: dist_figs/12_step_*.png
    - type: image
      glob: dist_figs/13_step_*.png
    - type: image
      glob: dist_figs/23_step_*.png
    - type: image
      glob: dist_figs/123_step_*.png
    - type: image
      glob: dist_figs/11_step_*.png
    - type: image
      glob: dist_figs/22_step_*.png
    - type: image
      glob: dist_figs/33_step_*.png
    - type: image
      glob: dist_figs/111_step_*.png
    - type: image
      glob: dist_figs/222_step_*.png
    - type: image
      glob: dist_figs/333_step_*.png
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
    - yKeys: ["d12_dist_l1/mean", "d13_dist_l1/mean", "d23_dist_l1/mean"]
      xKey: step
    - yKey: d123_dist_l1/mean
      xKey: step
    - yKeys: ["d11_dist_l1/mean", "d22_dist_l1/mean", "d33_dist_l1/mean"]
      xKey: step
    - yKeys: ["d111_dist_l1/mean", "d222_dist_l1/mean", "d333_dist_l1/mean"]
      xKey: step
    - yKey: mixture_dist_l1/mean
      xKey: step
    - yKey: steps_per_sec/mean
      xKey: step
    """, ".charts.yml", dedent=True)

    model_1 = load_model(Args.run_path_1, Args.device)
    model_2 = load_model(Args.run_path_2, Args.device)
    model_3 = load_model(Args.run_path_3, Args.device)

    fwd_logits_fn_1 = get_fwd_logits_fn(model_1, Args.horizon, Args.ndim, Args.device)
    logp_1 = compute_exact_logp(fwd_logits_fn_1, Args.horizon, Args.ndim, Args.device)
    fwd_logits_fn_2 = get_fwd_logits_fn(model_2, Args.horizon, Args.ndim, Args.device)
    logp_2 = compute_exact_logp(fwd_logits_fn_2, Args.horizon, Args.ndim, Args.device)
    fwd_logits_fn_3 = get_fwd_logits_fn(model_3, Args.horizon, Args.ndim, Args.device)
    logp_3 = compute_exact_logp(fwd_logits_fn_3, Args.horizon, Args.ndim, Args.device)

    logp_12_gt = logp_1 + logp_2 - \
                  1.0 * torch.logsumexp(torch.stack([logp_1, logp_2, logp_3], dim=0), dim=0)
    logp_12_gt = torch.log_softmax(logp_12_gt, dim=0)
    distr_12_gt = torch.exp(logp_12_gt)

    logp_13_gt = logp_1 + logp_3 - \
                 1.0 * torch.logsumexp(torch.stack([logp_1, logp_2, logp_3], dim=0), dim=0)
    logp_13_gt = torch.log_softmax(logp_13_gt, dim=0)
    distr_13_gt = torch.exp(logp_13_gt)

    logp_23_gt = logp_2 + logp_3 - \
                 1.0 * torch.logsumexp(torch.stack([logp_1, logp_2, logp_3], dim=0), dim=0)
    logp_23_gt = torch.log_softmax(logp_23_gt, dim=0)
    distr_23_gt = torch.exp(logp_23_gt)

    logp_123_gt = logp_1 + logp_2 + logp_3 - \
                  2.0 * torch.logsumexp(torch.stack([logp_1, logp_2, logp_3], dim=0), dim=0)
    logp_123_gt = torch.log_softmax(logp_123_gt, dim=0)
    distr_123_gt = torch.exp(logp_123_gt)

    logp_11_gt = 2.0 * logp_1 - \
                 1.0 * torch.logsumexp(torch.stack([logp_1, logp_2, logp_3], dim=0), dim=0)
    logp_11_gt = torch.log_softmax(logp_11_gt, dim=0)
    distr_11_gt = torch.exp(logp_11_gt)

    logp_22_gt = 2.0 * logp_2 - \
                 1.0 * torch.logsumexp(torch.stack([logp_1, logp_2, logp_3], dim=0), dim=0)
    logp_22_gt = torch.log_softmax(logp_22_gt, dim=0)
    distr_22_gt = torch.exp(logp_22_gt)

    logp_33_gt = 2.0 * logp_3 - \
                 1.0 * torch.logsumexp(torch.stack([logp_1, logp_2, logp_3], dim=0), dim=0)
    logp_33_gt = torch.log_softmax(logp_33_gt, dim=0)
    distr_33_gt = torch.exp(logp_33_gt)

    logp_111_gt = 3.0 * logp_1 - \
                  2.0 * torch.logsumexp(torch.stack([logp_1, logp_2, logp_3], dim=0), dim=0)
    logp_111_gt = torch.log_softmax(logp_111_gt, dim=0)
    distr_111_gt = torch.exp(logp_111_gt)

    logp_222_gt = 3.0 * logp_2 - \
                  2.0 * torch.logsumexp(torch.stack([logp_1, logp_2, logp_3], dim=0), dim=0)
    logp_222_gt = torch.log_softmax(logp_222_gt, dim=0)
    distr_222_gt = torch.exp(logp_222_gt)

    logp_333_gt = 3.0 * logp_3 - \
                  2.0 * torch.logsumexp(torch.stack([logp_1, logp_2, logp_3], dim=0), dim=0)
    logp_333_gt = torch.log_softmax(logp_333_gt, dim=0)
    distr_333_gt = torch.exp(logp_333_gt)


    logp_mixture_gt = torch.logsumexp(torch.stack([logp_1, logp_2, logp_3], dim=0), dim=0) - np.log(3)
    logp_mixture_gt = torch.log_softmax(logp_mixture_gt, dim=0)
    distr_mixture_gt = torch.exp(logp_mixture_gt)

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

    plot_distr('dist_figs/p1.png', logp_1.exp().detach().cpu().numpy(), 'P1')
    plot_distr('dist_figs/p2.png', logp_2.exp().detach().cpu().numpy(), 'P2')
    plot_distr('dist_figs/p3.png', logp_3.exp().detach().cpu().numpy(), 'P3')

    plot_distr('dist_figs/gt_12.png', logp_12_gt.exp().detach().cpu().numpy(), 'Ground Truth\n12')
    plot_distr('dist_figs/gt_13.png', logp_13_gt.exp().detach().cpu().numpy(), 'Ground Truth\n13')
    plot_distr('dist_figs/gt_23.png', logp_23_gt.exp().detach().cpu().numpy(), 'Ground Truth\n23')
    plot_distr('dist_figs/gt_123.png', logp_123_gt.exp().detach().cpu().numpy(), 'Ground Truth\n123')

    plot_distr('dist_figs/gt_11.png', logp_11_gt.exp().detach().cpu().numpy(), 'Ground Truth\n11')
    plot_distr('dist_figs/gt_22.png', logp_22_gt.exp().detach().cpu().numpy(), 'Ground Truth\n22')
    plot_distr('dist_figs/gt_33.png', logp_33_gt.exp().detach().cpu().numpy(), 'Ground Truth\n33')
    plot_distr('dist_figs/gt_111.png', logp_111_gt.exp().detach().cpu().numpy(), 'Ground Truth\n111')
    plot_distr('dist_figs/gt_222.png', logp_222_gt.exp().detach().cpu().numpy(), 'Ground Truth\n222')
    plot_distr('dist_figs/gt_333.png', logp_333_gt.exp().detach().cpu().numpy(), 'Ground Truth\n333')

    plot_distr('dist_figs/gt_mixture.png', logp_mixture_gt.exp().detach().cpu().numpy(), 'Ground Truth\nMixture')

    def save(cls, target_cls, opt, suffix='_last'):
        logger.torch_save({
            'cls': cls.state_dict(),
            'target_cls': target_cls.state_dict(),
            'opt': opt.state_dict(),
        }, f'checkpoints/model{suffix}.pt')


    cls = Joint3YClassifier(Args.horizon, Args.ndim, Args.num_hidden, Args.num_layers)

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
        x_3, trajectories_3 = sample_from_model(model_3, Args.horizon, Args.ndim, Args.batch_size, Args.device,
                                                return_trajectories=True)

        # compute terminal loss
        x_term = torch.cat([x_1, x_2, x_3], dim=0)

        enc_term = toin(x_term, Args.horizon).to(Args.device)
        logprobs_term = cls(enc_term, torch.ones(enc_term.shape[0], device=Args.device))

        logprobs_term_1 = logprobs_term[:x_1.shape[0]]
        logprobs_term_2 = logprobs_term[x_1.shape[0]:x_1.shape[0] + x_2.shape[0]]
        logprobs_term_3 = logprobs_term[x_1.shape[0] + x_2.shape[0]:]

        loss_1_term = -torch.mean(torch.logsumexp(logprobs_term_1, dim=(1, 2))[:, 0])  # -log P(y=1|x)
        loss_2_term = -torch.mean(torch.logsumexp(logprobs_term_2, dim=(1, 2))[:, 1])  # -log P(y=2|x)
        loss_3_term = -torch.mean(torch.logsumexp(logprobs_term_3, dim=(1, 2))[:, 2])  # -log P(y=3|x)

        loss_term = (loss_1_term + loss_2_term + loss_3_term) / 3.0


        # compute non-terminal loss

        s_1 = torch.cat(trajectories_1, dim=0)
        s_2 = torch.cat(trajectories_2, dim=0)
        s_3 = torch.cat(trajectories_3, dim=0)
        s_non_term = torch.cat([s_1, s_2, s_3], dim=0)
        enc_non_term = toin(s_non_term, Args.horizon).to(Args.device)
        traj_lens = [traj.shape[0] for traj in trajectories_1 + trajectories_2 + trajectories_3]
        traj_lens = torch.tensor(traj_lens, device=Args.device)
        traj_ind = torch.arange(0, traj_lens.shape[0], device=Args.device)
        traj_ind = traj_ind.repeat_interleave(traj_lens)

        with torch.no_grad():
            logprobs_term_ema = target_cls(enc_term, torch.ones(enc_term.shape[0], device=Args.device))

            p_x_y2_eq_1 = torch.sum(logprobs_term_ema.exp(), dim=(1, 2))[:, 0]
            p_x_y2_eq_2 = torch.sum(logprobs_term_ema.exp(), dim=(1, 2))[:, 1]
            p_x_y2_eq_3 = torch.sum(logprobs_term_ema.exp(), dim=(1, 2))[:, 2]

        logprobs_non_term = cls(enc_non_term, torch.zeros(enc_non_term.shape[0], device=Args.device))

        w_s_y2_eq_1 = p_x_y2_eq_1[traj_ind]
        w_s_y2_eq_2 = p_x_y2_eq_2[traj_ind]
        w_s_y2_eq_3 = p_x_y2_eq_3[traj_ind]

        w_s_yprobs = torch.stack([w_s_y2_eq_1, w_s_y2_eq_2, w_s_y2_eq_3], dim=1)
        w_s_yyprobs = w_s_yprobs[:, :, None] * w_s_yprobs[:, None, :]


        w_mat = torch.zeros((s_non_term.shape[0], 3, 3, 3), device=Args.device)
        # set y1 = 0
        w_mat[:s_1.shape[0], 0, :, :] = 1.0
        # set y1 = 1
        w_mat[s_1.shape[0]:s_1.shape[0] + s_2.shape[0], 1, :, :] = 1.0
        # set y1 = 2
        w_mat[s_1.shape[0] + s_2.shape[0]:, 2, :, :] = 1.0

        w_mat[:, :, :, :] *= w_s_yyprobs[:, None, :, :]

        loss_non_term = -torch.sum(w_mat * logprobs_non_term) / (3 * Args.batch_size)

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
            fwd_logits_12_fn = get_joint_guided_fwd_logits_fn(model_1, model_2, model_3, cls,
                                                              Args.horizon, Args.ndim,
                                                              Args.device, y1=1, y2=2, y3=None)
            logp_12_model = compute_exact_logp(fwd_logits_12_fn, Args.horizon, Args.ndim, Args.device)
            distr_12_model = logp_12_model.exp()
            assert (distr_12_model.sum() - 1.0).abs() < 1e-4

            logger.store_metrics(d12_dist_l1=np.abs(distr_12_model - distr_12_gt).sum())
            plot_distr(f'dist_figs/12_step_{step:08d}.png', distr_12_model, f'Generated distribution at step {step}')

            fwd_logits_13_fn = get_joint_guided_fwd_logits_fn(model_1, model_2, model_3, cls,
                                                              Args.horizon, Args.ndim,
                                                              Args.device, y1=1, y2=None, y3=3)
            logp_13_model = compute_exact_logp(fwd_logits_13_fn, Args.horizon, Args.ndim, Args.device)
            distr_13_model = logp_13_model.exp()
            assert (distr_13_model.sum() - 1.0).abs() < 1e-4

            logger.store_metrics(d13_dist_l1=np.abs(distr_13_model - distr_13_gt).sum())
            plot_distr(f'dist_figs/13_step_{step:08d}.png', distr_13_model, f'Generated distribution at step {step}')

            fwd_logits_23_fn = get_joint_guided_fwd_logits_fn(model_1, model_2, model_3, cls,
                                                              Args.horizon, Args.ndim,
                                                              Args.device, y1=None, y2=2, y3=3)
            logp_23_model = compute_exact_logp(fwd_logits_23_fn, Args.horizon, Args.ndim, Args.device)
            distr_23_model = logp_23_model.exp()
            assert (distr_23_model.sum() - 1.0).abs() < 1e-4

            logger.store_metrics(d23_dist_l1=np.abs(distr_23_model - distr_23_gt).sum())
            plot_distr(f'dist_figs/23_step_{step:08d}.png', distr_23_model, f'Generated distribution at step {step}')

            fwd_logits_123_fn = get_joint_guided_fwd_logits_fn(model_1, model_2, model_3, cls,
                                                               Args.horizon, Args.ndim,
                                                               Args.device, y1=1, y2=2, y3=3)
            logp_123_model = compute_exact_logp(fwd_logits_123_fn, Args.horizon, Args.ndim, Args.device)
            distr_123_model = logp_123_model.exp()
            assert (distr_123_model.sum() - 1.0).abs() < 1e-4

            logger.store_metrics(d123_dist_l1=np.abs(distr_123_model - distr_123_gt).sum())
            plot_distr(f'dist_figs/123_step_{step:08d}.png', distr_123_model, f'Generated distribution at step {step}')

            fwd_logits_11_fn = get_joint_guided_fwd_logits_fn(model_1, model_2, model_3, cls,
                                                              Args.horizon, Args.ndim,
                                                              Args.device, y1=1, y2=1, y3=None)
            logp_11_model = compute_exact_logp(fwd_logits_11_fn, Args.horizon, Args.ndim, Args.device)
            distr_11_model = logp_11_model.exp()
            assert (distr_11_model.sum() - 1.0).abs() < 1e-4

            logger.store_metrics(d11_dist_l1=np.abs(distr_11_model - distr_11_gt).sum())
            plot_distr(f'dist_figs/11_step_{step:08d}.png', distr_11_model, f'Generated distribution at step {step}')

            fwd_logits_22_fn = get_joint_guided_fwd_logits_fn(model_1, model_2, model_3, cls,
                                                              Args.horizon, Args.ndim,
                                                              Args.device, y1=2, y2=None, y3=2)
            logp_22_model = compute_exact_logp(fwd_logits_22_fn, Args.horizon, Args.ndim, Args.device)
            distr_22_model = logp_22_model.exp()
            assert (distr_22_model.sum() - 1.0).abs() < 1e-4

            logger.store_metrics(d22_dist_l1=np.abs(distr_22_model - distr_22_gt).sum())
            plot_distr(f'dist_figs/22_step_{step:08d}.png', distr_22_model, f'Generated distribution at step {step}')

            fwd_logits_33_fn = get_joint_guided_fwd_logits_fn(model_1, model_2, model_3, cls,
                                                              Args.horizon, Args.ndim,
                                                              Args.device, y1=None, y2=3, y3=3)
            logp_33_model = compute_exact_logp(fwd_logits_33_fn, Args.horizon, Args.ndim, Args.device)
            distr_33_model = logp_33_model.exp()
            assert (distr_33_model.sum() - 1.0).abs() < 1e-4

            logger.store_metrics(d33_dist_l1=np.abs(distr_33_model - distr_33_gt).sum())
            plot_distr(f'dist_figs/33_step_{step:08d}.png', distr_33_model, f'Generated distribution at step {step}')

            fwd_logits_111_fn = get_joint_guided_fwd_logits_fn(model_1, model_2, model_3, cls,
                                                               Args.horizon, Args.ndim,
                                                               Args.device, y1=1, y2=1, y3=1)
            logp_111_model = compute_exact_logp(fwd_logits_111_fn, Args.horizon, Args.ndim, Args.device)
            distr_111_model = logp_111_model.exp()
            assert (distr_111_model.sum() - 1.0).abs() < 1e-4

            logger.store_metrics(d111_dist_l1=np.abs(distr_111_model - distr_111_gt).sum())
            plot_distr(f'dist_figs/111_step_{step:08d}.png', distr_111_model, f'Generated distribution at step {step}')

            fwd_logits_222_fn = get_joint_guided_fwd_logits_fn(model_1, model_2, model_3, cls,
                                                                Args.horizon, Args.ndim,
                                                                Args.device, y1=2, y2=2, y3=2)
            logp_222_model = compute_exact_logp(fwd_logits_222_fn, Args.horizon, Args.ndim, Args.device)
            distr_222_model = logp_222_model.exp()
            assert (distr_222_model.sum() - 1.0).abs() < 1e-4

            logger.store_metrics(d222_dist_l1=np.abs(distr_222_model - distr_222_gt).sum())
            plot_distr(f'dist_figs/222_step_{step:08d}.png', distr_222_model, f'Generated distribution at step {step}')

            fwd_logits_333_fn = get_joint_guided_fwd_logits_fn(model_1, model_2, model_3, cls,
                                                                Args.horizon, Args.ndim,
                                                                Args.device, y1=3, y2=3, y3=3)
            logp_333_model = compute_exact_logp(fwd_logits_333_fn, Args.horizon, Args.ndim, Args.device)
            distr_333_model = logp_333_model.exp()
            assert (distr_333_model.sum() - 1.0).abs() < 1e-4

            logger.store_metrics(d333_dist_l1=np.abs(distr_333_model - distr_333_gt).sum())
            plot_distr(f'dist_figs/333_step_{step:08d}.png', distr_333_model, f'Generated distribution at step {step}')


            fwd_logits_mixture_fn = get_joint_guided_fwd_logits_fn(model_1, model_2, model_3, cls,
                                                                   Args.horizon, Args.ndim,
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
