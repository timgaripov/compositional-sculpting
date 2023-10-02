import math
import random

import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from params_proto import ParamsProto

class Eval(ParamsProto, prefix='eval'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = 100

    model_path_1 = None
    model_path_2 = None

    beta_1 = None  # None means maximal beta
    beta_2 = None  # None means maximal beta

    alpha = 0.5

    cls_path = None
    cls_y1 = 1  # y1 label for classifier-guidance {1, 2}
    cls_y2 = 2  # y2 label for classifier-guidance {1, 2}
    just_mixture = False

    batch_size = 75
    cls_max_batch_size = 4_000
    cls_num_workers = 8
    num_samples = 5000
    save_every = 500

    objectives = ['seh', 'sa']
    limits = [[-0.2, 1.2], [0.4, 0.95]]


from gflownet.fragment.mogfn import Args as ModelTrainingArgs, Trainer, SEHMOOTask, SEHSOOTask
from gflownet.fragment.train_joint_cls_param_onestep_beta import Args as ClsTrainingArgs
from gflownet.models.graph_transformer import GraphTransformerJointClassifierParam

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(run_path, device):
    from ml_logger import ML_Logger
    loader = ML_Logger(prefix=run_path)

    params_dict = loader.read_params('Args')
    ModelTrainingArgs.trajectory_balance._update(**params_dict['trajectory_balance'])
    params_dict.pop('trajectory_balance')
    ModelTrainingArgs._update(**params_dict)

    trainer = Trainer(ModelTrainingArgs, device, setup_ml_logger=False)

    assert isinstance(trainer.task, SEHSOOTask)
    assert ModelTrainingArgs.temperature_sample_dist != 'constant'

    trainer.model.to(device)
    saved_state = loader.torch_load('checkpoints/model_state.pt', map_location=device)

    trainer.model.load_state_dict(saved_state['models_state_dict'][0])

    return trainer, trainer.model


def load_cls(cls_path, model_trainer, device):
    from ml_logger import ML_Logger
    loader = ML_Logger(prefix=cls_path)

    params_dict = loader.read_params('Args')
    ClsTrainingArgs._update(**params_dict)

    cls = GraphTransformerJointClassifierParam(model_trainer.ctx, num_cond=1,
                                               num_emb=ClsTrainingArgs.num_emb,
                                               num_layers=ClsTrainingArgs.num_layers)
    cls.to(device)
    saved_state = loader.torch_load('checkpoints/model_last.pt', map_location=device)
    cls.load_state_dict(saved_state['cls'])

    return cls


class FixedCondInfoSampler(object):
    def __init__(self, task, beta):
        self.task = task
        self.beta = float(beta)

    def sample_conditional_information(self, batch_size):
        beta = None
        if self.beta is not None:
            beta = torch.full((batch_size,), self.beta, dtype=torch.float32)
        cond_dict = self.task.encode_conditional_information_custom_beta(beta, batch_size)

        return cond_dict


class ParamClsWrapper(object):
    def __init__(self, param_cls, alpha):
        self.param_cls = param_cls
        self.logit_alpha_v = math.log(alpha) - math.log(1.0 - alpha)

    def __call__(self, batch, terminal_tensor):
        logit_alpha_tensor = torch.full_like(terminal_tensor, self.logit_alpha_v)
        return self.param_cls(batch, logit_alpha_tensor, terminal_tensor)


def main(**deps):
    Eval._update(deps)
    set_seed(Eval.seed)

    from ml_logger import logger
    logger.log_params(Eval=vars(Eval))

    logger.log_text("""
        charts:
        - yKey: samples_per_sec/mean
          xKey: num_samples
        - type: image
          glob: dist_figs/samples_*.png
        """, ".charts.yml", dedent=True)

    model_trainer_1, model_1 = load_model(Eval.model_path_1, Eval.device)
    model_trainer_2, model_2 = load_model(Eval.model_path_2, Eval.device)

    param_cls = load_cls(Eval.cls_path, model_trainer_1, Eval.device)
    cls_wrapped = ParamClsWrapper(param_cls, Eval.alpha)

    cond_sampler_1 = FixedCondInfoSampler(model_trainer_1.task, Eval.beta_1)
    cond_sampler_2 = FixedCondInfoSampler(model_trainer_2.task, Eval.beta_2)

    def wrap_model(model):
        model.to(Eval.device)
        return model, Eval.device

    eval_task = SEHMOOTask(Eval.objectives, [],
                           temperature_sample_dist='constant', temperature_parameters=1.0,
                           num_thermometer_dim=1, rng=None,
                           wrap_model=wrap_model)

    def dummy_graph_cls(batch, terminal_tensor):
        return batch.x.new_zeros(batch.num_graphs)

    sns.set_style('whitegrid')

    def save_distplot(path, flat_rewards, flat_reward_names,
                      limits=((0, 1), (0, 1)), title=''):
        plt.figure(figsize=(10, 8))
        joint_data = {name: flat_rewards[:, i] for i, name in enumerate(flat_reward_names)}
        g = sns.jointplot(joint_data, x=flat_reward_names[0], y=flat_reward_names[1],
                          kind='scatter', s=14, alpha=0.12,
                          xlim=limits[0], ylim=limits[1],
                          marginal_ticks=True,
                          marginal_kws=dict(stat='density'))
        g.plot_joint(sns.kdeplot, zorder=0,
                     n_levels=8, bw_adjust=0.95,
                     alpha=0.5, lw=2)
        g.plot_marginals(sns.kdeplot, fill=True, alpha=0.5)
        plt.xlabel(flat_reward_names[0], fontsize=16)
        plt.ylabel(flat_reward_names[1], fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.title(title, fontsize=24, y=1.2)

        logger.savefig(path)
        plt.close()

    num_invalid = 0
    num_total = 0
    generated_mols = []
    flat_rewards = np.empty((0, len(Eval.objectives)), dtype=np.float32)
    flat_reward_names = eval_task.flat_reward_names()
    vis_id = [flat_reward_names.index(name) for name in Eval.objectives[:2]]

    last_saved_samples = 0
    logger.start('last_saved')
    progress = tqdm.tqdm(total=Eval.num_samples, desc='Generating molecules')
    while len(generated_mols) < Eval.num_samples:
        n = min(Eval.batch_size, Eval.num_samples - len(generated_mols))

        cond_info_1 = cond_sampler_1.sample_conditional_information(n)['encoding']
        cond_info_2 = cond_sampler_2.sample_conditional_information(n)['encoding']

        data = model_trainer_1.algo.graph_sampler.sample_from_model_guided_joint_beta(
            model_1,
            model_2,
            cls_wrapped,
            n,
            cond_info_1.to(Eval.device),
            cond_info_2.to(Eval.device),
            dev=Eval.device,
            random_action_prob=0.0,
            cls_y1=Eval.cls_y1,
            cls_y2=Eval.cls_y2,
            just_mixture=Eval.just_mixture,
            cls_max_batch_size=Eval.cls_max_batch_size,
            cls_num_workers=Eval.cls_num_workers
        )
        valid_idcs = [i for i in range(len(data)) if data[i]['is_valid']]
        batch_mols = [model_trainer_1.ctx.graph_to_mol(data[i]['result']) for i in valid_idcs]

        batch_flat_rewards, is_valid = eval_task.compute_flat_rewards(batch_mols)
        batch_flat_rewards = batch_flat_rewards.cpu().numpy()
        is_valid = is_valid.cpu().numpy()
        valid_reward_idcs = np.where(is_valid)[0]
        generated_mols.extend([batch_mols[i] for i in valid_reward_idcs])
        flat_rewards = np.concatenate((flat_rewards, batch_flat_rewards[valid_reward_idcs]), axis=0)

        num_generated = len(generated_mols)
        num_invalid += n - valid_reward_idcs.shape[0]
        num_total += n

        if (num_generated - last_saved_samples >= Eval.save_every) or (num_generated >= Eval.num_samples):
            samples_per_sec = (num_generated - last_saved_samples) / logger.split('last_saved')
            last_saved_samples = num_generated

            logger.store_metrics(samples_per_sec=samples_per_sec)
            logger.log_metrics_summary(key_values={'num_samples': len(generated_mols)})

            save_distplot(f'dist_figs/samples_{num_generated:08d}.png',
                          flat_rewards[:, vis_id],
                          [flat_reward_names[i] for i in vis_id],
                          limits=Eval.limits,
                          title=f'Generated {num_generated} molecules\n')
            logger.save_pkl({
                'generated_mols': generated_mols,
                'flat_rewards': flat_rewards,
                'flat_reward_names': flat_reward_names,
                'num_generated': num_generated,
                'num_invalid': num_invalid,
                'num_total': num_total,
            }, path=f'results.pkl', append=False)

        progress.update(valid_reward_idcs.shape[0])

    print(f'Generated {len(generated_mols)} valid molecules')
    print(f'Number of invalid molecules: {num_invalid}/{num_total}')


if __name__ == '__main__':
    from ml_logger import instr
    thunk = instr(main)
    thunk()
