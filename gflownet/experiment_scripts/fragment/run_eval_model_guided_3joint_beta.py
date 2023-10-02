if __name__ == '__main__':

    import itertools
    from pathlib import Path
    from gflownet.fragment.eval_model_guided_3joint_beta import main, Eval
    from gflownet.experiment_scripts import RUN
    from params_proto.hyper import Sweep
    import jaynes
    from ml_logger import instr

    jaynes.config('local', verbose=False)

    with Sweep(RUN, Eval) as sweep:
        with sweep.set:
            Eval.num_samples = 5_000
            Eval.batch_size = 75
            Eval.cls_max_batch_size = 4_000
            Eval.cls_num_workers = 8
            Eval.save_every = 500

            Eval.seed = 100
            Eval.objectives = ['seh', 'sa', 'qed', 'mw']

            with sweep.chain:
                with sweep.product:
                    Eval.just_mixture = [False]
                    with sweep.zip:
                        # HM, diff(P^1, P^2), diff(P^2, P^1)
                        Eval.cls_y1 = [1, 1, 2, 3, 1, 1, 2, 1, 2, 3]
                        Eval.cls_y2 = [2, 1, 2, 3, 2, 3, 3, 1, 2, 3]
                        Eval.cls_y3 = [3, 1, 2, 3, None, None, None, None, None, None]

                    with sweep.zip:
                        Eval.model_path_1 = [
                            '',  # <seh gflownet run path>
                        ]
                        Eval.beta_1 = [
                            32.0,
                        ]
                        Eval.model_path_2 = [
                            '',  # <sa gflownet run path>
                        ]
                        Eval.beta_2 = [
                            32.0,
                        ]
                        Eval.model_path_3 = [
                            '',  # <qed gflownet run path>
                        ]
                        Eval.beta_3 = [
                            32.0,
                        ]

                        Eval.cls_path = [
                            '',  # <classifier seh_beta_32 vs sa_beta_32 vs qed_beta_32 run path>
                        ]

                with sweep.product:
                    Eval.just_mixture = [True]
                    with sweep.zip:
                        Eval.model_path_1 = [
                            '',  # <seh gflownet run path>
                        ]
                        Eval.beta_1 = [
                            32.0,
                        ]
                        Eval.model_path_2 = [
                            '',  # <sa gflownet run path>
                        ]
                        Eval.beta_2 = [
                            32.0,
                        ]
                        Eval.model_path_3 = [
                            '',  # <qed gflownet run path>
                        ]
                        Eval.beta_3 = [
                            32.0,
                        ]

                        Eval.cls_path = [
                            '',  # <classifier seh_beta_32 vs sa_beta_32 vs qed_beta_32 run path>
                        ]

    @sweep.each
    def tail(RUN, Eval):
        comb_tag = f'y{Eval.cls_y1}{Eval.cls_y2}{Eval.cls_y3}'
        if Eval.just_mixture:
            comb_tag = f'mixture'
        RUN.job_name = (f'{{now:%H.%M.%S}}/'
                        f'cls_{Eval.cls_path.split("/")[-1]}'
                        f'_{comb_tag}')

    sweep.save(f'{Path(__file__).stem}.jsonl')

    # truncate iterator to only 1 item for demonstration
    # comment this line out for to run all experiments
    sweep = itertools.islice(sweep, 1)

    # gpus_to_use = [0, 1, 2, 3]
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
