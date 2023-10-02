# the code here is mostly copied from this tutorial: https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from params_proto import PrefixProto

from diffusion_chaining.ddpm import diffusion_coeff
from diffusion_chaining.ddpm_sampler import pc_sampler, marginal_prob_std


class DDPM_comp(PrefixProto, cli=False):
    dist_1 = None
    dist_2 = None
    gen_1 = None
    gen_2 = None
    model_path = "checkpoints/model_last.pt"

    alpha = 20.0
    sigma = 25.0
    snr = 0.16
    # 250 steps does not affect the results
    n_steps = 250
    sample_batch_size = 8 * 8

    seed = 100

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"


class Sculptor(nn.Module):
    def __init__(self, score_model1, score_model2, classifier, y_1, y_2, guidance_scale=1.0):
        super().__init__()
        self.score_model1 = score_model1
        self.score_model2 = score_model2
        self.classifier = classifier
        self.y_1 = y_1
        self.y_2 = y_2
        self.guidance_scale = guidance_scale

        self.input_channels = score_model1.input_channels

    def classifier_grad(self, x, t):
        x_tmp = torch.clone(x).requires_grad_(True).to(DDPM_comp.device)
        t.requires_grad_(False)
        cls_logprobs_x_t = self.classifier(x_tmp, t)

        grd = torch.zeros((x.shape[0], 2, 2), device=DDPM_comp.device)  # same shape as cls_logprobs_x_t
        grd[:, self.y_1, self.y_2] = 1.0  # column of Jacobian to compute
        cls_logprobs_x_t.backward(gradient=grd, retain_graph=True)
        grad = x_tmp.grad
        grad.requires_grad_(False)

        return grad

    def forward(self, x, t):
        with torch.enable_grad():
            cls_grad = self.classifier_grad(x, t)
        with torch.no_grad():
            score_1 = self.score_model1(x, t)
            score_2 = self.score_model2(x, t)

            cls_logprobs_x_t = self.classifier(x, t)

            # calculate p(y_1 = 1 | x_t) and p(y_1 = 2 | x_t)
            p_y1_eq_1_x_t = torch.sum(torch.exp(cls_logprobs_x_t), dim=2)[:, 0]
            p_y1_eq_2_x_t = torch.sum(torch.exp(cls_logprobs_x_t), dim=2)[:, 1]

            mixture_score = torch.mul(score_1, p_y1_eq_1_x_t.view(-1, 1, 1, 1)) + torch.mul(score_2, p_y1_eq_2_x_t.view(-1, 1, 1, 1))
            # print(torch.mean(torch.norm(mixture_score, dim=[2,3])), torch.mean(torch.norm(cls_grad, dim=[2,3])))
            composition_score = mixture_score + self.guidance_scale * cls_grad
            return composition_score


def composite_factory(dist: str, guidance_scale, device=None):
    """Factory for making composites.

    Not used here, but used for chaining. - Ge

    :param dist: "m1-m2", "m2-m1", "m1xm2"
    :param path_template:
    :param device:
    :return:
    """
    from ml_logger import logger

    # sort the
    if "-" in dist:
        dist_1, dist_2 = dist.split("-")

        if dist_1 < dist_2:
            yy = 0, 0
        else:
            dist_2, dist_1 = dist_1, dist_2
            yy = 1, 1

    elif "x" in dist:
        dist_1, dist_2 = dist.split("x")
        yy = 0, 1

    gen_1_path = f"/toy-diffusion/toy-diffusion/neurips/ddpm/base/{dist_1}/100"
    gen_2_path = f"/toy-diffusion/toy-diffusion/neurips/ddpm/base/{dist_2}/100"
    clfr_path = f"/toy-diffusion/toy-diffusion/neurips/ddpm/bcomp/{dist_1}-{dist_2}/100"

    gen_1 = logger.torch_load(gen_1_path, "checkpoints/model_last.pt", map_location=device)
    gen_1.requires_grad_(False)

    gen_2 = logger.torch_load(gen_2_path, "checkpoints/model_last.pt", map_location=device)
    gen_2.requires_grad_(False)

    clfr_2ord = logger.torch_load(clfr_path, "checkpoints/model_last.pt", map_location=device)
    clfr_2ord.requires_grad_(False)

    composed_model = Sculptor(gen_1, gen_2, clfr_2ord, *yy, guidance_scale=guidance_scale)

    return composed_model


# if __name__ == "__main__":
#     cm = composite_factory("m1xm2", guidance_scale=20.0, device="cuda")
#     exit()


def I_sample(model, title):
    from ml_logger import logger

    ## Generate samples using the specified sampler.
    samples = pc_sampler(
        model,
        partial(marginal_prob_std, sigma=DDPM_comp.sigma),
        partial(diffusion_coeff, sigma=DDPM_comp.sigma),
        DDPM_comp.sample_batch_size,
        device=DDPM_comp.device,
    )

    ## Sample visualization.
    samples = samples.clamp(0.0, 1.0)

    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt

    sample_grid = make_grid(samples, nrow=int(np.sqrt(DDPM_comp.sample_batch_size)))

    plt.figure(figsize=(4, 4))
    plt.axis("off")
    # plt.title(title)
    # fmt: off
    logger.log_text(f"""
    - type: image
      glob: "{title}.png"
    """, ".charts.yml", dedent=True, overwrite=False, )

    logger.save_image(sample_grid.permute(1, 2, 0).cpu().numpy(), f"{title}.png")
    # plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0.0, vmax=1.0)
    # plt.tight_layout(pad=0)
    # logger.savefig(f"{title}.png", dpi=180, bbox_inches="tight")
    # plt.show()


def main(**deps):
    from ml_logger import logger
    from diffusion_chaining.models.util import set_seed

    DDPM_comp._update(deps)
    logger.log_params(DDPM_comp=vars(DDPM_comp))
    # fmt: off
    logger.log_text("""
    charts:
    """, ".charts.yml", dedent=True, overwrite=True)
    print(logger.get_dash_url())
    set_seed(DDPM_comp.seed)

    gen_1 = logger.torch_load(DDPM_comp.gen_1, DDPM_comp.model_path, map_location=DDPM_comp.device)
    gen_1.requires_grad_(False)

    gen_2 = logger.torch_load(DDPM_comp.gen_2, DDPM_comp.model_path, map_location=DDPM_comp.device)
    gen_2.requires_grad_(False)

    clfr_2ord = logger.torch_load(DDPM_comp.clfr, DDPM_comp.model_path, map_location=DDPM_comp.device)
    clfr_2ord.requires_grad_(False)

    y1, y2 = 0, 0
    composed_model_y11 = Sculptor(gen_1, gen_2, clfr_2ord, y1, y2, DDPM_comp.alpha)

    y1, y2 = 0, 1
    composed_model_y12 = Sculptor(gen_1, gen_2, clfr_2ord, y1, y2, DDPM_comp.alpha)

    y1, y2 = 1, 1
    composed_model_y22 = Sculptor(gen_1, gen_2, clfr_2ord, y1, y2, DDPM_comp.alpha)

    I_sample(gen_1, f"{DDPM_comp.dist_1}")
    I_sample(gen_2, f"{DDPM_comp.dist_2}")

    I_sample(composed_model_y11, f"{DDPM_comp.dist_1}-{DDPM_comp.dist_2}")
    I_sample(composed_model_y12, f"{DDPM_comp.dist_1}x{DDPM_comp.dist_2}")
    I_sample(composed_model_y22, f"{DDPM_comp.dist_2}-{DDPM_comp.dist_1}")


# if __name__ == "__main__":
#     DDPM_comp.dist_1 = "m1"
#     DDPM_comp.dist_2 = "m2"
#     DDPM_comp.gen_1 = "/toy-diffusion/toy-diffusion/neurips/ddpm/base/m1/100"
#     DDPM_comp.gen_2 = "/toy-diffusion/toy-diffusion/neurips/ddpm/base/m2/100"
#
#     DDPM_comp.clfr = "/toy-diffusion/toy-diffusion/neurips/ddpm/bcomp/m1-m2/100"
#     main()
#     exit()

if __name__ == "__main__":
    import jaynes
    from params_proto.hyper import Sweep
    from ml_logger.job import RUN, instr

    with Sweep(DDPM_comp, RUN).product as sweep:
        with sweep.zip:
            # DDPM_comp.dist_1 = ["M1", "M_odd", "M_even"]
            # DDPM_comp.dist_2 = ["M2", "M_three", "M_three"]
            DDPM_comp.dist_1 = ["M_a", "M_b", "M_a"]
            DDPM_comp.dist_2 = ["M_b", "M_c", "M_c"]

        DDPM_comp.seed = [100, 200, 300]

    def tail(D: DDPM_comp, RUN):
        d1, d2 = D.dist_1.lower(), D.dist_2.lower()

        D.gen_1 = f"/toy-diffusion/toy-diffusion/neurips/ddpm/base/{d1}/{DDPM_comp.seed}"
        D.gen_2 = f"/toy-diffusion/toy-diffusion/neurips/ddpm/base/{d2}/{DDPM_comp.seed}"

        D.clfr = f"/toy-diffusion/toy-diffusion/neurips/ddpm/bcomp/{d1}-{d2}/{DDPM_comp.seed}"

        RUN.prefix = f"toy-diffusion/toy-diffusion/neurips/ddpm/bcomp_samples/{d1}-{d2}/{DDPM_comp.seed}"

    sweep.each(tail)

    gpus_to_use = [0, 1, 2, 3]

    jaynes.config("local")
    for i, deps in enumerate(sweep):

        RUN.CUDA_VISIBLE_DEVICES = str(gpus_to_use[i % len(gpus_to_use)])
        jaynes.config("local")
        thunk = instr(main, **deps, __diff=False)
        jaynes.run(thunk)

    jaynes.listen()
    print("All Done!")
