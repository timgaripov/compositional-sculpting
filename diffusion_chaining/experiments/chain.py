from pathlib import Path

from ml_logger.job import RUN
from params_proto.hyper import Sweep

from diffusion_chaining.chain import DDPM_chain

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


def tail(DDPM_chain, RUN):
    d1 = DDPM_chain.dist_1.lower()
    d2 = DDPM_chain.dist_2.lower()

    DDPM_chain.gen_1 = f"/toy-diffusion/toy-diffusion/neurips/ddpm/base/{d1}/{DDPM_chain.seed}"

    RUN.prefix = f"toy-diffusion/toy-diffusion/neurips/ddpm/chain/{d1}-{d2}/{DDPM_chain.seed}"


sweep.each(tail).save(f"{Path(__file__).stem}.jsonl")
