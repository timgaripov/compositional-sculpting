if __name__ == '__main__':

    import itertools
    from pathlib import Path
    from gflownet.fragment.eval_model_beta import main, Eval
    from gflownet.experiment_scripts import RUN
    from params_proto.hyper import Sweep
    import jaynes
    from ml_logger import instr

    jaynes.config('local', verbose=False)

    with Sweep(RUN, Eval) as sweep:
        with sweep.set:
            Eval.num_samples = 5_000
            Eval.batch_size = 100
            Eval.save_every = 1_000

            with sweep.product:
                with sweep.zip:
                    Eval.model_path = [
                        '',  # <seh gflownet run path>
                        '',  # <sa gflownet run path>
                        '',  # <qed gflownet run path>
                    ]
                    Eval.objectives = [
                        ['seh', 'sa', 'qed'],
                        ['seh', 'sa', 'qed'],
                        ['seh', 'qed', 'sa'],
                    ]
                    Eval.limits = [
                        [[-0.2, 1.2], [0.4, 0.95]],
                        [[-0.2, 1.2], [0.4, 0.95]],
                        [[-0.2, 1.2], [-0.1, 1.1]],
                    ]

                # None means maximal beta (96)
                Eval.beta = [None, 32.0]
                Eval.seed = [100]

    @sweep.each
    def tail(RUN, Eval):
        beta_str = f'beta_{int(Eval.beta)}' if Eval.beta else 'beta_96'
        RUN.job_name = f"{{now:%H.%M.%S}}/{Eval.model_path.split('/')[-1]}_{beta_str}"

    sweep.save(f'{Path(__file__).stem}.jsonl')

    # truncate iterator to only 1 item for demonstration
    # comment this line out for to run all experiments
    sweep = itertools.islice(sweep, 1)

    # gpus_to_use = [0, 1, 2]
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
