if __name__ == '__main__':

    from pathlib import Path
    from gflownet.grid.train_grid_cls_2dist_param import main, Args
    from gflownet.experiment_scripts import RUN
    from params_proto.hyper import Sweep
    import jaynes
    from ml_logger import instr

    jaynes.config('local', verbose=False)

    with Sweep(RUN, Args) as sweep:
        with sweep.set:
            Args.num_training_steps = 15_000
            Args.loss_non_term_weight_steps = 3_000
            Args.target_network_ema = 0.995
            Args.batch_size = 64

            Args.run_path_1 = ''  # <path to gflownet run 1>
            Args.run_path_2 = ''  # <path to gflownet run 2>
            Args.seed = 100


    @sweep.each
    def tail(RUN, Args):
        RUN.job_name = f"{{now:%H.%M.%S}}/{Args.run_path_1.split('/')[-1]}_vs_{Args.run_path_2.split('/')[-1]}"

    sweep.save(f'{Path(__file__).stem}.jsonl')

    for kwargs in sweep:
        thunk = instr(main, **kwargs)
        jaynes.run(thunk)
    jaynes.listen()
