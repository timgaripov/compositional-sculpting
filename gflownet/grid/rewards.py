import math
import numpy as np

class LogRewardFns:
    # x in [-1.0, 1.0]
    @staticmethod
    def corners(x):
        ax = x.abs()
        r = (ax > 0.5).prod(-1) * 0.5 + ((ax < 0.8) * (ax > 0.6)).prod(-1) * 2
        log_r = (r + 1e-3).log()
        return log_r

    @staticmethod
    def currin(x):
        x_0 = x[..., 0] / 2 + 0.5
        x_1 = x[..., 1] / 2 + 0.5
        factor1 = 1 - np.exp(- 1 / (2 * x_1 + 1e-10))
        numer = 2300 * x_0 ** 3 + 1900 * x_0 ** 2 + 2092 * x_0 + 60
        denom = 100 * x_0 ** 3 + 500 * x_0 ** 2 + 4 * x_0 + 20
        r = factor1 * numer / denom / 13.77  # Dividing by the max to help normalize
        log_r = (r + 1e-8).log()
        return log_r

    @staticmethod
    def branin(x):
        x_0 = 15 * (x[..., 0] / 2 + 0.5) - 5
        x_1 = 15 * (x[..., 1] / 2 + 0.5)
        t1 = (x_1 - 5.1 / (4 * np.pi ** 2) * x_0 ** 2
              + 5 / np.pi * x_0 - 6)
        t2 = 10 * (1 - 1 / (8 * np.pi)) * np.cos(x_0)
        r = 1 - (t1 ** 2 + t2 + 10) / 308.13  # Dividing by the max to help normalize
        log_r = (r + 1e-8).log()
        return log_r


    @staticmethod
    def shubert(x):
        # my best attempt at reproducing the shubert function
        # http://profesores.elo.utfsm.cl/~tarredondo/info/soft-comp/functions/node28.html
        w = 2.3
        u_1 = -7.15
        u_2 = -7.15
        x_1 = (u_1 - w) + (x[..., 0] / 2.0 + 0.5) * w
        x_2 = (u_2 - w) + (x[..., 1] / 2.0 + 0.5) * w

        mn = -186.6157949555621
        mx = 210.27662470796076

        cosine_sum_1 = 0
        cosine_sum_2 = 0
        for i in range(1, 6):
            cosine_sum_1 = cosine_sum_1 + i * (x_1 * (i + 1) + i).cos()
            cosine_sum_2 = cosine_sum_2 + i * (x_2 * (i + 1) + i).cos()

        r = (cosine_sum_1 * cosine_sum_2 - mn) / (mx - mn)
        log_r = (r + 1e-3).log()

        return log_r

    @staticmethod
    def symmetric_shubert(x):
        # symmetrized version of the above
        # makes probabilities of the modes more equal
        w = 2.3
        u_1 = -7.15
        u_2 = -7.15
        x_1 = (u_1 - w) + (x[..., 0].abs() / 2.0 + 0.5) * w
        x_2 = (u_2 - w) + (x[..., 1].abs() / 2.0 + 0.5) * w

        mn = -186.6157949555621
        mx = 210.27662470796076

        cosine_sum_1 = 0
        cosine_sum_2 = 0
        for i in range(1, 6):
            cosine_sum_1 = cosine_sum_1 + i * (x_1 * (i + 1) + i).cos()
            cosine_sum_2 = cosine_sum_2 + i * (x_2 * (i + 1) + i).cos()

        r = (cosine_sum_1 * cosine_sum_2 - mn) / (mx - mn)
        log_r = (r + 1e-3).log()

        return log_r

    @staticmethod
    def diag_sigmoid(x):
        r = (x.sum(-1) * 5).sigmoid()
        log_r = (r + 1e-5).log()
        return log_r

    @staticmethod
    def circle1(x):
        x_1 = x[..., 0]
        x_2 = x[..., 1]
        r = 0.6
        h = 0.3

        center = [-h, 0.0]
        dist = ((x_1 - center[0]) ** 2 + (x_2 - center[1]) ** 2) ** 0.5

        in_mask = dist < r
        out_mask = dist >= r

        normal_offsets = [
            [0.0, 0.0]
        ]
        normal_std = 0.3
        normal_densities = []
        mixture_density = None


        for offset in normal_offsets:
            ncenter = [center[0] + offset[0], center[1] + offset[1]]
            density = (-0.5 * ((x_1 - ncenter[0]) ** 2 + (x_2 - ncenter[1]) ** 2) / normal_std ** 2).exp() * \
                      1.0 / (2 * np.pi * (normal_std ** 2))
            normal_densities.append(density)

            if mixture_density is None:
                mixture_density = density / len(normal_offsets)
            else:
                mixture_density = mixture_density + density / len(normal_offsets)

        r = (mixture_density * 2.5 + 6.5) * in_mask.float()
        r += 0.1 * out_mask.float()

        log_r = (r + 1e-8).log()
        return log_r

    @staticmethod
    def circle2(x):
        x_1 = x[..., 0]
        x_2 = x[..., 1]
        r = 0.6
        h = 0.3

        center = [h * 0.5, h * math.sqrt(3) / 2.0]
        dist = ((x_1 - center[0]) ** 2 + (x_2 - center[1]) ** 2) ** 0.5

        in_mask = dist < r
        out_mask = dist >= r

        hm = 0.32

        normal_offsets = [
            [-hm * np.sqrt(3.0) / 2.0, 0.5 * hm],
            [hm * np.sqrt(3.0) / 2.0, -0.5 * hm],
        ]
        normal_std = 0.21
        normal_densities = []
        mixture_density = None


        for offset in normal_offsets:
            ncenter = [center[0] + offset[0], center[1] + offset[1]]
            density = (-0.5 * ((x_1 - ncenter[0]) ** 2 + (x_2 - ncenter[1]) ** 2) / normal_std ** 2).exp() * \
                      1.0 / (2 * np.pi * (normal_std ** 2))
            normal_densities.append(density)

            if mixture_density is None:
                mixture_density = density / len(normal_offsets)
            else:
                mixture_density = mixture_density + density / len(normal_offsets)

        r = (mixture_density * 3.5 + 10.5) * in_mask.float()
        r += 0.1 * out_mask.float()

        log_r = (r + 1e-8).log()
        return log_r

    @staticmethod
    def circle3(x):
        x_1 = x[..., 0]
        x_2 = x[..., 1]
        r = 0.6
        h = 0.3

        center = [0.5 * h, -math.sqrt(3) / 2.0 * h]
        dist = ((x_1 - center[0]) ** 2 + (x_2 - center[1]) ** 2) ** 0.5

        in_mask = dist < r
        out_mask = dist >= r

        hm = 0.32

        normal_offsets = [
            [-hm, 0.0],
            [0.5 * hm, math.sqrt(3) / 2.0 * hm],
            [0.5 * hm, -math.sqrt(3) / 2.0 * hm],
        ]
        normal_std = 0.18
        normal_densities = []
        mixture_density = None


        for offset in normal_offsets:
            ncenter = [center[0] + offset[0], center[1] + offset[1]]
            density = (-0.5 * ((x_1 - ncenter[0]) ** 2 + (x_2 - ncenter[1]) ** 2) / normal_std ** 2).exp() * \
                      1.0 / (2 * np.pi * (normal_std ** 2))
            normal_densities.append(density)

            if mixture_density is None:
                mixture_density = density / len(normal_offsets)
            else:
                mixture_density = mixture_density + density / len(normal_offsets)

        r = (mixture_density * 2.5 + 5.5) * in_mask.float()
        r += 0.1 * out_mask.float()

        log_r = (r + 1e-8).log()
        return log_r


if __name__ == '__main__':
    import torch
    import matplotlib.pyplot as plt
    import seaborn as sns

    ndim = 2
    horizon = 32

    reward_temperature = 1.0

    for reward_name in ['circle1', 'circle2', 'circle3']:

        base_log_reward_fn = getattr(LogRewardFns, reward_name)

        def log_reward(z):
            x_scaled = z / (horizon - 1) * 2 - 1
            base_log_r = base_log_reward_fn(x_scaled)

            return base_log_r / reward_temperature


        pos = torch.zeros((horizon,) * ndim + (ndim,))
        for i in range(ndim):
            pos_i = torch.linspace(0, horizon - 1, horizon)
            for _ in range(i):
                pos_i = pos_i.unsqueeze(1)
            pos[..., i] = pos_i

        truelr = log_reward(pos)
        print('total reward', truelr.view(-1).logsumexp(0))
        true_dist = truelr.flatten().softmax(0).cpu().numpy()

        cmap = sns.color_palette("Blues", as_cmap=True)

        def plot_distr(distribution, title):
            distribution_2d = distribution.reshape(horizon, horizon).T

            vmax = distribution_2d.max()
            vmin = 0.0 - 0.05 * vmax

            fig = plt.figure(figsize=(10, 10))
            plt.imshow(distribution_2d, cmap=cmap,
                       interpolation='nearest', vmin=vmin, vmax=vmax)
            plt.colorbar()
            plt.title(title, fontsize=24)

            plt.show()
            plt.close()

        plot_distr(true_dist, f'Ground truth')
