from pathlib import Path

from ml_logger.job import RUN
from params_proto.hyper import Sweep

from diffusion_chaining.bcomp import DDPM_comp

with Sweep(DDPM_comp, RUN) as sweep:
    # there is no num_steps for inference
    # DDPM.n

    with sweep.product:
        with sweep.zip:
            DDPM_comp.dist_1 = ["m1", "m_even", "m_odd", "m_a", "m_b", "m_a"]
            DDPM_comp.dist_2 = ["m2", "m_three", "m_three", "m_b", "m_c", "m_c"]

        DDPM_comp.seed = [100, 200, 300]


def tail(DDPM_comp, RUN):
    DDPM_comp.gen_1 = f"/toy-diffusion/toy-diffusion/neurips/ddpm/base/{DDPM_comp.dist_1}/{DDPM_comp.seed}"
    DDPM_comp.gen_2 = f"/toy-diffusion/toy-diffusion/neurips/ddpm/base/{DDPM_comp.dist_2}/{DDPM_comp.seed}"

    RUN.prefix = f"toy-diffusion/toy-diffusion/neurips/ddpm/bcomp/{DDPM_comp.dist_1}-{DDPM_comp.dist_2}/{DDPM_comp.seed}"


sweep.each(tail).save(f"{Path(__file__).stem}.jsonl")
