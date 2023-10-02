if __name__ == '__main__':

    import itertools
    from pathlib import Path
    from gflownet.grid.train_grid import main, Args
    from gflownet.experiment_scripts import RUN
    from params_proto.hyper import Sweep
    import jaynes
    from ml_logger import instr

    jaynes.config('local', verbose=False)

    with Sweep(RUN, Args) as sweep:
        with sweep.set:
            Args.num_training_steps = 20_000
            Args.batch_size = 16

            with sweep.product:
                with sweep.zip:
                    Args.reward_name = ['circle1', 'circle2', 'circle3']
                    Args.reward_temperature = [1.0, 1.0, 1.0]
                Args.uniform_pb = [True]
                Args.seed = [100, 200, 300]

    @sweep.each
    def tail(RUN, Args):
        uniform_pb_flag = 'uniform_pb' if Args.uniform_pb else 'learned_pb'
        RUN.job_name = f"{{now:%H.%M.%S}}/{Args.reward_name}_{uniform_pb_flag}_{Args.seed}"

    sweep.save(f'{Path(__file__).stem}.jsonl')

    # truncate iterator to only 1 item for demonstration
    # comment this line out for to run all experiments
    sweep = itertools.islice(sweep, 1)

    for kwargs in sweep:
        thunk = instr(main, **kwargs)
        jaynes.run(thunk)
    jaynes.listen()
