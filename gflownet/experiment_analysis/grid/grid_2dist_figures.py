import math
import os
import shutil

import numpy as np

import torch

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

from gflownet.grid.train_grid import Args as ModelArgs, make_model, compute_exact_logp, get_fwd_logits_fn
from gflownet.grid.train_grid_cls_2dist_param import Args as ClsArgs, JointYClassifierParam, get_joint_guided_fwd_logits_fn


def load_model(run_path, device):
    from ml_logger import ML_Logger
    loader = ML_Logger(prefix=run_path)
    ModelArgs._update(**loader.read_params('Args'))

    model, _ = make_model(ModelArgs.horizon, ModelArgs.ndim, ModelArgs.num_hidden, ModelArgs.num_layers)
    model.to(device)
    saved_state = loader.torch_load('checkpoints/model_last.pt', map_location=device)
    model.load_state_dict(saved_state['model'])

    return model


def load_classifier(run_path, device):
    from ml_logger import ML_Logger
    loader = ML_Logger(prefix=run_path)
    ClsArgs._update(**loader.read_params('Args'))

    cls = JointYClassifierParam(ClsArgs.horizon, ClsArgs.ndim, ClsArgs.num_hidden, ClsArgs.num_layers)
    cls.to(device)
    saved_state = loader.torch_load('checkpoints/model_last.pt', map_location=device)
    cls.load_state_dict(saved_state['cls'])

    return cls, ClsArgs.horizon, ClsArgs.ndim


def create_figure_row(dist_list, horizon, title_list, contours_list,
                      gamma_list,
                      n_pad_cols=0,
                      dir=None, name=None, show=False, cbar=False):
    ncols = len(dist_list) + 2 * n_pad_cols
    if ncols > 1:
        fig, axes = plt.subplots(1, ncols, figsize=(ncols * 2.8 + 2, 3.5))
    else:
        fig = plt.figure(figsize=(ncols * 2.8 + 2, 3.5))
        axes = [fig.gca()]


    cmap = sns.color_palette("Blues", as_cmap=True)
    contour_color = sns.color_palette('tab10')[3]


    im = None
    for i, (dist, title, contours, gamma) in enumerate(zip(dist_list, title_list, contours_list, gamma_list)):
        ax_id = i + n_pad_cols
        plt.sca(axes[ax_id])
        vmax = np.percentile(dist, 99.5)
        vmin = 0.0 - 0.05 * vmax

        dist_2d = dist.reshape(horizon, horizon).T

        im = plt.imshow(dist_2d, cmap=cmap,
                        norm=colors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax),
                        interpolation='nearest')
        for ci, contour in enumerate(contours):
            if contour is None:
                continue
            plt.plot(contour[:, 0], contour[:, 1], '-',
                     c=contour_color, linewidth=0.6,
                     zorder=10)
        plt.axis('off')

    for i in range(ncols):
        if n_pad_cols <= i < ncols - n_pad_cols:
            continue
        plt.sca(axes[i])
        plt.axis('off')

    fig.subplots_adjust(left=0.02, right=0.82, hspace=0.4, wspace=0.18)

    if cbar:
        cbar_ax = fig.add_axes([0.86, 0.15, 0.09, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.formatter.set_powerlimits((0, 0))
        cbar.formatter.set_useMathText(True)
        cbar_ax.set_axis_off()

    if dir is not None:
        plt.savefig(os.path.join(dir, f'{name}.pdf'), bbox_inches='tight')
        plt.savefig(os.path.join(dir, f'{name}.png'), bbox_inches='tight', dpi=300)

    if show:
        plt.show()
    plt.close()


if __name__ == '__main__':
    model_path_1 = ''  # <path to gflownet run 1>
    model_path_2 = ''  # <path to gflownet run 2>
    cls_path = ''  # <path to classifier run>


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_1 = load_model(model_path_1, device)
    model_2 = load_model(model_path_2, device)
    cls, horizon, ndim = load_classifier(cls_path, device)

    results_dir = os.path.basename(__file__)[:-3]
    shutil.rmtree(results_dir, ignore_errors=True)
    os.makedirs(results_dir, exist_ok=True)

    fwd_logits_fn_1 = get_fwd_logits_fn(model_1, horizon, ndim, device)
    logp_1 = compute_exact_logp(fwd_logits_fn_1, horizon, ndim, device)
    fwd_logits_fn_2 = get_fwd_logits_fn(model_2, horizon, ndim, device)
    logp_2 = compute_exact_logp(fwd_logits_fn_2, horizon, ndim, device)


    name_list = ['p1', 'p2']
    gamma_list = [1.0, 1.0]
    dist_list = [logp_1.exp(), logp_2.exp()]

    for alpha_str in ['005', '050', '095']:
        name_list.append(f'hm_{alpha_str}')

        alpha = float(alpha_str) / 100.0
        logit_alpha = math.log(alpha) - math.log(1.0 - alpha)
        fwd_logits_hm_fn = get_joint_guided_fwd_logits_fn(model_1, model_2, cls, horizon, ndim,
                                                          device, y1=1, y2=2, logit_alpha=logit_alpha)
        logp_hm_model = compute_exact_logp(fwd_logits_hm_fn, horizon, ndim, device)
        distr_hm_model = logp_hm_model.exp()
        dist_list.append(distr_hm_model)
        gamma_list.append(1.0)

    for alpha_str in ['005', '050', '095']:
        name_list.append(f'diff_12_{alpha_str}')

        alpha = float(alpha_str) / 100.0
        logit_alpha = math.log(alpha) - math.log(1.0 - alpha)
        fwd_logits_diff_fn = get_joint_guided_fwd_logits_fn(model_1, model_2, cls, horizon, ndim,
                                                            device, y1=1, y2=1, logit_alpha=logit_alpha)
        logp_diff_model = compute_exact_logp(fwd_logits_diff_fn, horizon, ndim, device)
        distr_diff_model = logp_diff_model.exp()
        dist_list.append(distr_diff_model)
        gamma_list.append(1.5 if alpha_str == '050' else 1.0)

    for alpha_str in ['005', '050', '095']:
        name_list.append(f'diff_21_{alpha_str}')

        alpha = float(alpha_str) / 100.0
        logit_alpha = math.log(alpha) - math.log(1.0 - alpha)
        fwd_logits_diff_fn = get_joint_guided_fwd_logits_fn(model_1, model_2, cls, horizon, ndim,
                                                            device, y1=2, y2=2, logit_alpha=logit_alpha)
        logp_diff_model = compute_exact_logp(fwd_logits_diff_fn, horizon, ndim, device)
        distr_diff_model = logp_diff_model.exp()
        dist_list.append(distr_diff_model)
        gamma_list.append(1.5 if alpha_str == '050' else 1.0)


    for i, (name, dist, gamma) in enumerate(zip(name_list, dist_list, gamma_list)):
        create_figure_row([dist], horizon,
                          title_list=[''], contours_list=[[]],
                          gamma_list=[gamma],
                          n_pad_cols=0, dir=results_dir,
                          name=name, show=True, cbar=i == len(dist_list) - 1)
