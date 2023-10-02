if __name__ == '__main__':

    from pathlib import Path
    from gflownet.fragment.train_3joint_cls_onestep_beta import main, Args
    from gflownet.experiment_scripts import RUN
    from params_proto.hyper import Sweep
    import jaynes
    from ml_logger import instr

    jaynes.config('local', verbose=False)

    with Sweep(RUN, Args) as sweep:
        with sweep.set:
            Args.num_training_steps = 15_000
            Args.log_every = 250
            Args.save_every = 1_000

            Args.batch_size = 8

            Args.loss_non_term_weight_steps = 4_000
            Args.target_network_ema = 0.995

            with sweep.zip:
                Args.run_path_1 = [
                    '',  # <seh gflownet run path>
                ]
                Args.beta_1 = [
                    32.0,
                ]
                Args.run_path_2 = [
                    '',  # <sa gflownet run path>
                ]
                Args.beta_2 = [
                    32.0,
                ]
                Args.run_path_3 = [
                    '',  # <qed gflownet run path>
                ]
                Args.beta_3 = [
                    32.0,
                ]
            Args.seed = 100


    @sweep.each
    def tail(RUN, Args):
        def cond_str(beta):
            beta_str = f'beta_{int(beta)}' if beta else 'beta_96'
            return f'{beta_str}'
        cond_str_1 = cond_str(Args.beta_1)
        cond_str_2 = cond_str(Args.beta_2)
        cond_str_3 = cond_str(Args.beta_3)
        RUN.job_name = (f"{{now:%H.%M.%S}}/{Args.run_path_1.split('/')[-1]}_{cond_str_1}"
                        f"_vs_{Args.run_path_2.split('/')[-1]}_{cond_str_2}"
                        f"_vs_{Args.run_path_3.split('/')[-1]}_{cond_str_3}")

    sweep.save(f'{Path(__file__).stem}.jsonl')

    # gpus_to_use = [0]
    gpus_to_use = [None]

    gpu_id = 0
    for i, kwargs in enumerate(sweep):
        RUN.CUDA_VISIBLE_DEVICES = gpus_to_use[gpu_id % len(gpus_to_use)]
        if RUN.CUDA_VISIBLE_DEVICES is not None:
            RUN.CUDA_VISIBLE_DEVICES = str(RUN.CUDA_VISIBLE_DEVICES)
        thunk = instr(main, **kwargs)
        jaynes.run(thunk)
        gpu_id += 1
    jaynes.listen()
