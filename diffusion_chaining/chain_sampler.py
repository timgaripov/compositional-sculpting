# the code here is mostly copied from this tutorial: https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing
from functools import partial

import numpy as np
import torch
from params_proto import PrefixProto

from diffusion_chaining.bcomp_sampler import composite_factory, Sculptor
from diffusion_chaining.ddpm import diffusion_coeff
from diffusion_chaining.ddpm_sampler import pc_sampler, marginal_prob_std


class DDPM_chain(PrefixProto, cli=False):
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


def I_sample(model, title):
    from ml_logger import logger

    ## Generate samples using the specified sampler.
    samples = pc_sampler(
        model,
        partial(marginal_prob_std, sigma=DDPM_chain.sigma),
        partial(diffusion_coeff, sigma=DDPM_chain.sigma),
        DDPM_chain.sample_batch_size,
        device=DDPM_chain.device,
    )

    ## Sample visualization.
    samples = samples.clamp(0.0, 1.0)

    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt

    sample_grid = make_grid(samples, nrow=int(np.sqrt(DDPM_chain.sample_batch_size)))

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

    DDPM_chain._update(deps)
    logger.log_params(DDPM_comp=vars(DDPM_chain))
    # fmt: off
    logger.log_text("""
    charts:
    """, ".charts.yml", dedent=True, overwrite=True)
    print(logger.get_dash_url())
    set_seed(DDPM_chain.seed)

    gen_1 = logger.torch_load(DDPM_chain.gen_1, DDPM_chain.model_path, map_location=DDPM_chain.device)
    gen_1.requires_grad_(False)

    gen_2 = composite_factory(DDPM_chain.dist_2, guidance_scale=20.0, device=DDPM_chain.device)

    clfr_2ord = logger.torch_load(DDPM_chain.clfr, DDPM_chain.model_path, map_location=DDPM_chain.device)
    clfr_2ord.requires_grad_(False)

    y1, y2 = 0, 0
    composed_model_y11 = Sculptor(gen_1, gen_2, clfr_2ord, y1, y2, DDPM_chain.alpha)

    y1, y2 = 0, 1
    composed_model_y12 = Sculptor(gen_1, gen_2, clfr_2ord, y1, y2, DDPM_chain.alpha)

    y1, y2 = 1, 1
    composed_model_y22 = Sculptor(gen_1, gen_2, clfr_2ord, y1, y2, DDPM_chain.alpha)

    I_sample(gen_1, f"{DDPM_chain.dist_1}")
    I_sample(gen_2, f"{DDPM_chain.dist_2}")

    I_sample(composed_model_y11, f"{DDPM_chain.dist_1}-({DDPM_chain.dist_2})")
    I_sample(composed_model_y12, f"{DDPM_chain.dist_1}x({DDPM_chain.dist_2})")
    I_sample(composed_model_y22, f"{DDPM_chain.dist_2}-({DDPM_chain.dist_1})")


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

    with Sweep(DDPM_chain, RUN).product as sweep:
        with sweep.chain:
            with sweep.product:
                DDPM_chain.dist_1 = ["m_a"]
                DDPM_chain.dist_2 = ["m_bxm_c", "m_b-m_c", "m_c-m_b"]
            with sweep.product:
                DDPM_chain.dist_1 = ["m_b"]
                DDPM_chain.dist_2 = ["m_axm_c", "m_a-m_c", "m_c-m_a"]
            with sweep.product:
                DDPM_chain.dist_1 = ["m_c"]
                DDPM_chain.dist_2 = ["m_axm_b", "m_a-m_b", "m_b-m_a"]

        DDPM_chain.seed = [100, 200, 300]

    def tail(D: DDPM_chain, RUN):

        d1, d2 = D.dist_1.lower(), D.dist_2.lower()
        D.gen_1 = f"/toy-diffusion/toy-diffusion/neurips/ddpm/base/{d1}/{DDPM_chain.seed}"
        D.clfr = f"/toy-diffusion/toy-diffusion/neurips/ddpm/chain/{d1}-{d2}/{DDPM_chain.seed}"

        RUN.prefix = f"toy-diffusion/toy-diffusion/neurips/ddpm/chain_samples/{d1}-{d2}/{DDPM_chain.seed}"

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
