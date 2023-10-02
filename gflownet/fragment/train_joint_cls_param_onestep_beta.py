import copy
import random

import numpy as np
import torch
import torch_geometric.data as gd

from params_proto import ParamsProto


class Args(ParamsProto, prefix='classifier-2dist'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = 100

    run_path_1 = None
    run_path_2 = None

    beta_1 = None
    beta_2 = None

    logit_alpha_range = [-5.5, 5.5]

    batch_size = 8

    num_emb = 128
    num_layers = 4

    num_data_loader_workers = 4

    num_training_steps = 15_000
    target_network_ema = 0.995
    learning_rate = 0.001
    weight_decay = 1e-6
    loss_non_term_weight_steps = 4_000

    log_every = 250
    save_every = 1000


from gflownet.data.sampling_iterator import SimpleSamplingIterator
from gflownet.envs.graph_building_env import GraphActionCategorical
from gflownet.fragment.mogfn import Args as TrainingArgs, Trainer, SEHSOOTask
from gflownet.models.graph_transformer import GraphTransformerJointClassifierParam
from gflownet.utils.multiprocessing_proxy import wrap_model_mp

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(run_path, device):
    from ml_logger import ML_Logger
    loader = ML_Logger(prefix=run_path)

    params_dict = loader.read_params('Args')
    TrainingArgs.trajectory_balance._update(**params_dict['trajectory_balance'])
    params_dict.pop('trajectory_balance')
    TrainingArgs._update(**params_dict)

    trainer = Trainer(TrainingArgs, device, setup_ml_logger=False)

    assert isinstance(trainer.task, SEHSOOTask)
    assert TrainingArgs.temperature_sample_dist != 'constant'

    trainer.model.to(device)
    saved_state = loader.torch_load('checkpoints/model_state.pt', map_location=device)

    trainer.model.load_state_dict(saved_state['models_state_dict'][0])

    return trainer, trainer.model


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


def disable_grad(model):
    for p in model.parameters():
        p.requires_grad = False


def build_loader(model, cond_info_sampler, graph_sampler, batch_size, result_only, device, num_workers):
    iterator_device = device
    wrapped_model = model
    if num_workers > 0:
        wrapped_model = wrap_model_mp(model, num_workers, cast_types=(gd.Batch, GraphActionCategorical))
        iterator_device = torch.device('cpu')

    iterator = SimpleSamplingIterator(wrapped_model, cond_info_sampler, graph_sampler, batch_size,
                                      result_only=result_only, device=iterator_device)

    return torch.utils.data.DataLoader(iterator, batch_size=None,
                                       num_workers=num_workers, persistent_workers=num_workers > 0)


def main(**deps):

    Args._update(deps)
    set_seed(Args.seed)

    from ml_logger import logger

    logger.log_params(Args=vars(Args))
    logger.log_text("""
        charts:
        - yKey: loss/mean
          xKey: step
        - yKey: loss_term/mean
          xKey: step
        - yKey: loss_non_term/mean
          xKey: step
        - yKey: loss_non_term_weight/mean
          xKey: step
        - yKeys: ["output_term_min/min", "output_term_max/max"]
          xKey: step
        - yKey: grad_norm/mean
          xKey: step
        - yKey: param_norm/mean
          xKey: step
        - yKey: steps_per_sec/mean
          xKey: step
        - yKeys: ["frac_invalid_1/mean", "frac_invalid_2/mean"]
          xKey: step
        """, ".charts.yml", dedent=True)


    trainer_1, model_1 = load_model(Args.run_path_1, Args.device)
    trainer_2, model_2 = load_model(Args.run_path_2, Args.device)
    disable_grad(model_1)
    disable_grad(model_2)

    cls = GraphTransformerJointClassifierParam(trainer_1.ctx, num_cond=1,
                                               num_emb=Args.num_emb,
                                               num_layers=Args.num_layers)

    target_cls = copy.deepcopy(cls)
    disable_grad(target_cls)

    cls.to(Args.device)
    target_cls.to(Args.device)

    cond_info_sampler_1 = FixedCondInfoSampler(trainer_1.task, Args.beta_1)
    cond_info_sampler_2 = FixedCondInfoSampler(trainer_2.task, Args.beta_2)
    graph_sampler_1 = trainer_1.algo.graph_sampler
    graph_sampler_2 = trainer_2.algo.graph_sampler

    loader_1 = build_loader(model_1, cond_info_sampler_1, graph_sampler_1, Args.batch_size,
                            result_only=False, device=Args.device,
                            num_workers=Args.num_data_loader_workers)
    loader_2 = build_loader(model_2, cond_info_sampler_2, graph_sampler_2, Args.batch_size,
                            result_only=False, device=Args.device,
                            num_workers=Args.num_data_loader_workers)

    data_iter_1 = iter(loader_1)
    data_iter_2 = iter(loader_2)

    opt = torch.optim.Adam(cls.parameters(),
                           lr=Args.learning_rate, weight_decay=Args.weight_decay)

    def save(cls, target_cls, opt, suffix='_last'):
        logger.torch_save({
            'cls': cls.state_dict(),
            'target_cls': target_cls.state_dict(),
            'opt': opt.state_dict(),
        }, f'checkpoints/model{suffix}.pt')

    early_checkpoint_flag = False

    logger.start('log_timer')
    timer_steps = 0

    for step in range(Args.num_training_steps):
        batch_1 = next(data_iter_1)
        batch_2 = next(data_iter_2)

        batch_1.to(Args.device)
        batch_2.to(Args.device)

        traj_lens_1 = batch_1.traj_lens
        x_ind_1 = torch.cumsum(traj_lens_1, dim=0) - 1
        s_mask_1 = torch.all(torch.arange(batch_1.num_graphs, device=Args.device)[:, None] != x_ind_1[None, :], dim=1)
        s_ind_1 = torch.nonzero(s_mask_1, as_tuple=True)[0]
        s_traj_ind_1 = torch.sum((s_ind_1[:, None] > x_ind_1[None, :]).long(), dim=1)

        batch_1_s = gd.Batch.from_data_list(batch_1.index_select(s_ind_1),
                                            follow_batch=['edge_index']).to(Args.device)
        batch_1_x = gd.Batch.from_data_list(batch_1.index_select(x_ind_1),
                                            follow_batch=['edge_index']).to(Args.device)

        traj_lens_2 = batch_2.traj_lens
        x_ind_2 = torch.cumsum(traj_lens_2, dim=0) - 1
        s_mask_2 = torch.all(torch.arange(batch_2.num_graphs, device=Args.device)[:, None] != x_ind_2[None, :], dim=1)
        s_ind_2 = torch.nonzero(s_mask_2, as_tuple=True)[0]
        s_traj_ind_2 = torch.sum((s_ind_2[:, None] > x_ind_2[None, :]).long(), dim=1)

        batch_2_s = gd.Batch.from_data_list(batch_2.index_select(s_ind_2),
                                            follow_batch=['edge_index']).to(Args.device)
        batch_2_x = gd.Batch.from_data_list(batch_2.index_select(x_ind_2),
                                            follow_batch=['edge_index']).to(Args.device)

        u_1 = torch.rand((batch_1_x.num_graphs, 1), device=Args.device)
        u_2 = torch.rand((batch_2_x.num_graphs, 1), device=Args.device)
        logit_alpha_1 = Args.logit_alpha_range[0] + (Args.logit_alpha_range[1] - Args.logit_alpha_range[0]) * u_1
        logit_alpha_2 = Args.logit_alpha_range[0] + (Args.logit_alpha_range[1] - Args.logit_alpha_range[0]) * u_2

        _, outputs_1_term = cls.get_outputs(
            batch_1_x,
            logit_alpha_1,
            torch.ones((batch_1_x.num_graphs, 1), device=Args.device))
        _, outputs_2_term = cls.get_outputs(
            batch_2_x,
            logit_alpha_2,
            torch.ones((batch_2_x.num_graphs, 1), device=Args.device))

        output_term_min = min(
            torch.min(outputs_1_term).item(),
            torch.min(outputs_2_term).item()
        )
        output_term_max = max(
            torch.max(outputs_1_term).item(),
            torch.max(outputs_2_term).item()
        )

        logp_1_x_y1_eq_1 = torch.nn.functional.logsigmoid(-outputs_1_term).squeeze()
        logp_2_x_y1_eq_2 = torch.nn.functional.logsigmoid(outputs_2_term).squeeze()

        loss_1_term = -torch.mean(logp_1_x_y1_eq_1)  # -log P(y1=1|x_1)
        loss_2_term = -torch.mean(logp_2_x_y1_eq_2)  # -log P(y1=2|x_2)
        loss_term = 0.5 * (loss_1_term + loss_2_term)

        # compute phase 2 loss
        with torch.no_grad():
            _, outputs_1_term_ema = target_cls.get_outputs(
                batch_1_x,
                logit_alpha_1,
                torch.ones((batch_1_x.num_graphs, 1), device=Args.device))
            _, outputs_2_term_ema = target_cls.get_outputs(
                batch_2_x,
                logit_alpha_2,
                torch.ones((batch_2_x.num_graphs, 1), device=Args.device))

            p_1_x_y2_eq_1 = torch.sigmoid(-(outputs_1_term_ema - logit_alpha_1)).squeeze()
            p_1_x_y2_eq_2 = torch.sigmoid(outputs_1_term_ema - logit_alpha_1).squeeze()
            p_2_x_y2_eq_1 = torch.sigmoid(-(outputs_2_term_ema - logit_alpha_2)).squeeze()
            p_2_x_y2_eq_2 = torch.sigmoid(outputs_2_term_ema - logit_alpha_2).squeeze()


        logprobs_1_non_term = cls(
            batch_1_s,
            logit_alpha_1[s_traj_ind_1],
            torch.zeros((batch_1_s.num_graphs, 1), device=Args.device))
        logprobs_2_non_term = cls(
            batch_2_s,
            logit_alpha_2[s_traj_ind_2],
            torch.zeros((batch_2_s.num_graphs, 1), device=Args.device))

        w_1_s_y2_eq_1 = p_1_x_y2_eq_1[s_traj_ind_1]
        w_1_s_y2_eq_2 = p_1_x_y2_eq_2[s_traj_ind_1]
        w_2_s_y2_eq_1 = p_2_x_y2_eq_1[s_traj_ind_2]
        w_2_s_y2_eq_2 = p_2_x_y2_eq_2[s_traj_ind_2]

        w_1_mat = torch.zeros((batch_1_s.num_graphs, 2, 2), device=Args.device)
        w_2_mat = torch.zeros((batch_2_s.num_graphs, 2, 2), device=Args.device)

        w_1_mat[:, 0, 0] = 1.0
        w_1_mat[:, 0, 1] = 1.0
        w_1_mat[:, :, 0] *= w_1_s_y2_eq_1[:, None]
        w_1_mat[:, :, 1] *= w_1_s_y2_eq_2[:, None]

        w_2_mat[:, 1, 0] = 1.0
        w_2_mat[:, 1, 1] = 1.0
        w_2_mat[:, :, 0] *= w_2_s_y2_eq_1[:, None]
        w_2_mat[:, :, 1] *= w_2_s_y2_eq_2[:, None]

        loss_1_non_term = -torch.sum(w_1_mat * logprobs_1_non_term) / batch_1_x.num_graphs
        loss_2_non_term = -torch.sum(w_2_mat * logprobs_2_non_term) / batch_2_x.num_graphs
        loss_non_term = 0.5 * (loss_1_non_term + loss_2_non_term)

        loss_non_term_weight = 1.0
        if Args.loss_non_term_weight_steps > 0:
            loss_non_term_weight = min(1.0, step / Args.loss_non_term_weight_steps)

        loss = loss_term + loss_non_term * loss_non_term_weight

        frac_invalid_1 = batch_1.num_invalid / Args.batch_size
        frac_invalid_2 = batch_2.num_invalid / Args.batch_size

        opt.zero_grad()
        loss.backward()
        opt.step()

        # update target network
        for a, b in zip(cls.parameters(), target_cls.parameters()):
            b.data.mul_(Args.target_network_ema).add_(a.data * (1 - Args.target_network_ema))

        timer_steps += 1
        grad_norm = sum([p.grad.detach().norm() ** 2 for p in cls.parameters()]) ** 0.5
        param_norm = sum([p.detach().norm() ** 2 for p in cls.parameters()]) ** 0.5
        logger.store_metrics({
            'loss': loss.item(),
            'grad_norm': grad_norm.item(),
            'param_norm': param_norm.item(),
            'loss_term': loss_term.item(),
            'loss_non_term': loss_non_term.item(),
            'loss_non_term_weight': loss_non_term_weight,
            'output_term_min': output_term_min,
            'output_term_max': output_term_max,
            'frac_invalid_1': frac_invalid_1,
            'frac_invalid_2': frac_invalid_2,
        })

        if (step % Args.save_every == 0) or (step == Args.num_training_steps - 1):
            save(cls, target_cls, opt)

            if (not early_checkpoint_flag) and (step >= 500):
                early_checkpoint_flag = True
                save(cls, target_cls, opt, f'_{step:08d}')

            if step % 4000 == 0:
                save(cls, target_cls, opt, f'_{step:08d}')

        if step % Args.log_every == 0:
            logger.store_metrics({
                'steps_per_sec': timer_steps / logger.split('log_timer')
            })
            timer_steps = 0
            logger.log_metrics_summary(key_values={'step': step},
                                       key_stats={'output_term_min': 'min_max',
                                                  'output_term_max': 'min_max',
                                                  })


if __name__ == '__main__':
    from ml_logger import instr
    thunk = instr(main)
    thunk()
