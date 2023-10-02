if __name__ == '__main__':

    import itertools
    from pathlib import Path
    from gflownet.fragment.mogfn import main, Args
    from gflownet.experiment_scripts import RUN
    from params_proto.hyper import Sweep
    import jaynes
    from ml_logger import instr

    jaynes.config('local', verbose=False)

    with Sweep(RUN, Args) as sweep:
        with sweep.set:
            Args.temperature_sample_dist = 'uniform'
            Args.temperature_dist_params = [0.0, 96.0]
            Args.num_thermometer_dim = 32
            Args.global_batch_size = 64
            Args.sampling_tau = 0.95
            Args.num_emb = 128
            Args.num_layers = 6

            Args.num_training_steps = 20_000

            Args.preference_type = 'seeded_single'
            Args.n_valid_repeats_per_pref = 128

            Args.num_data_loader_workers = 8

            with sweep.product:
                with sweep.zip:
                    Args.objectives = [['seh'], ['qed'], ['sa']]
                    Args.learning_rate = [0.0005, 0.0001, 0.0005]
                    Args.Z_learning_rate = [0.0005, 0.001, 0.0005]
                Args.seed = [100, 200, 300]

    @sweep.each
    def tail(RUN, Args):
        RUN.job_name = f"{{now:%H.%M.%S}}/{Args.objectives[0]}_{Args.seed}"

    sweep.save(f'{Path(__file__).stem}.jsonl')

    # truncate iterator to only 1 item for demonstration
    # comment this line out for to run all experiments
    sweep = itertools.islice(sweep, 1)

    # gpus_to_use = [0, 1, 2, 3]
    gpus_to_use = [None]

    gpu_id = 0
    for kwargs in sweep:
        RUN.CUDA_VISIBLE_DEVICES = gpus_to_use[gpu_id % len(gpus_to_use)]
        if RUN.CUDA_VISIBLE_DEVICES is not None:
            RUN.CUDA_VISIBLE_DEVICES = str(RUN.CUDA_VISIBLE_DEVICES)
        thunk = instr(main, **kwargs)
        jaynes.run(thunk)
        gpu_id += 1
    jaynes.listen()
