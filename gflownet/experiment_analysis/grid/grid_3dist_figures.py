import math
import os
import shutil

import numpy as np

import torch

import matplotlib.pyplot as plt
import seaborn as sns

from gflownet.grid.train_grid import Args as ModelArgs, make_model, compute_exact_logp, get_fwd_logits_fn
from gflownet.grid.train_grid_cls_3dist import Args as ClsArgs, Joint3YClassifier, get_joint_guided_fwd_logits_fn


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

    cls = Joint3YClassifier(ClsArgs.horizon, ClsArgs.ndim, ClsArgs.num_hidden, ClsArgs.num_layers)
    cls.to(device)
    saved_state = loader.torch_load('checkpoints/model_last.pt', map_location=device)
    cls.load_state_dict(saved_state['cls'])

    return cls, ClsArgs.horizon, ClsArgs.ndim


def create_figure_row(dist_list, horizon, title_list, contours_list, n_pad_cols=0,
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
    for i, (dist, title, contours) in enumerate(zip(dist_list, title_list, contours_list)):
        ax_id = i + n_pad_cols
        plt.sca(axes[ax_id])

        vmax = np.percentile(dist, 98.0)
        vmin = 0.0 - 0.05 * vmax

        dist_2d = dist.reshape(horizon, horizon).T

        im = plt.imshow(dist_2d, cmap=cmap,
                        interpolation='nearest', vmin=vmin, vmax=vmax)
        for ci, contour in enumerate(contours):
            if contour is None:
                continue
            plt.plot(contour[:, 0], contour[:, 1], '-',
                     c=contour_color, linewidth=0.8,
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
    model_path_3 = ''  # <path to gflownet run 3>
    cls_path = ''  # <path to classifier run>

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_1 = load_model(model_path_1, device)
    model_2 = load_model(model_path_2, device)
    model_3 = load_model(model_path_3, device)
    cls, horizon, ndim = load_classifier(cls_path, device)

    results_dir = os.path.basename(__file__)[:-3]
    shutil.rmtree(results_dir, ignore_errors=True)
    os.makedirs(results_dir, exist_ok=True)

    fwd_logits_fn_1 = get_fwd_logits_fn(model_1, horizon, ndim, device)
    logp_1 = compute_exact_logp(fwd_logits_fn_1, horizon, ndim, device)
    fwd_logits_fn_2 = get_fwd_logits_fn(model_2, horizon, ndim, device)
    logp_2 = compute_exact_logp(fwd_logits_fn_2, horizon, ndim, device)
    fwd_logits_fn_3 = get_fwd_logits_fn(model_3, horizon, ndim, device)
    logp_3 = compute_exact_logp(fwd_logits_fn_3, horizon, ndim, device)


    name_list = ['p1', 'p2', 'p3']
    dist_list = [logp_1.exp(), logp_2.exp(), logp_3.exp()]

    def get_circle_contour(center, radius):
        angles = np.linspace(0, 2 * np.pi, 100)
        x = center[0] + radius * np.cos(angles)
        y = center[1] + radius * np.sin(angles)
        return np.stack([x, y], axis=1)

    scale = 31 / 2.0

    h = 0.3 * scale
    r = 0.63 * scale

    circles = [
        get_circle_contour([15.5, 15.5 - h], r),
        get_circle_contour([15.5 + math.sqrt(3) / 2.0 * h, 15.5 + 0.5 * h], r),
        get_circle_contour([15.5 - math.sqrt(3) / 2.0 * h, 15.5 + 0.5 * h], r),
    ]


    ycombs = [
        [1, 2, None], [2, 3, None], [1, 3, None], [1, 2, 3],
        [1, 1, None], [2, 2, None], [3, 3, None],
        [1, 1, 1], [2, 2, 2], [3, 3, 3],
    ]
    for ycomb in ycombs:
        y1, y2, y3 = ycomb
        fwd_logits_fn = get_joint_guided_fwd_logits_fn(model_1, model_2, model_3, cls,
                                                       horizon, ndim,
                                                       device, y1=y1, y2=y2, y3=y3)
        logp_model = compute_exact_logp(fwd_logits_fn, horizon, ndim, device)
        distr_model = logp_model.exp()

        dist_list.append(distr_model)
        y_str = ''
        if y1 is not None:
            y_str += f'{y1}'
        if y2 is not None:
            y_str += f'{y2}'
        if y3 is not None:
            y_str += f'{y3}'
        name_list.append(f'cls_y{y_str}')

    for i, (name, dist) in enumerate(zip(name_list, dist_list)):
        create_figure_row([dist], horizon,
                          title_list=[''], contours_list=[circles],
                          n_pad_cols=0, dir=results_dir,
                          name=name, show=False, cbar=i == len(dist_list) - 1)
