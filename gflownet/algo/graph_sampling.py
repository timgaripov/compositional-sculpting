import copy
import time

import math
from typing import List

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import IterableDataset

from gflownet.envs.graph_building_env import GraphActionType


def extract_logprobs(logprobs, cls_y1, cls_y2, cls_y3):
    out = logprobs
    if cls_y3 is None:
        out = torch.logsumexp(out, dim=-1)
    else:
        out = out[:, :, :, cls_y3 - 1]

    if cls_y2 is None:
        out = torch.logsumexp(out, dim=-1)
    else:
        out = out[:, :, cls_y2 - 1]

    if cls_y1 is None:
        out = torch.logsumexp(out, dim=-1)
    else:
        out = out[:, cls_y1 - 1]

    return out

class SuccessorGraphDataset(IterableDataset):
    def __init__(self,
               env,
               ctx,
               graphs,
               torch_graphs,
               fwd_logits,
               fwd_masks,
               fwd_batch,
               fwd_slice,
               batch_ind_to_graph_ind,
               share_memory=False):
        super().__init__()
        self.env = env
        self.ctx = ctx
        self.graphs = graphs
        self.torch_graphs = [x.detach().cpu().clone() for x in torch_graphs]
        self.batch_ind_to_graph_ind = batch_ind_to_graph_ind
        self.fwd_batch = [x.detach().cpu().clone() for x in fwd_batch]
        self.fwd_slice = [x.detach().cpu().clone() for x in fwd_slice]

        broadcasted_masks = [
            torch.broadcast_tensors(logits, mask)[1]
            for logits, mask in zip(fwd_logits, fwd_masks)
        ]
        # masks are floats 0.0 or 1.0, using 0.5 threshold
        self.unmasked_indices = [
            torch.nonzero(mask > 0.5, as_tuple=True)
            for mask in broadcasted_masks
        ]
        self.unmasked_indices = [tuple(y.detach().cpu().clone() for y in x) for x in self.unmasked_indices]
        if share_memory:
            for x in self.torch_graphs:
                x.share_memory_()
            for x in self.fwd_batch:
                x.share_memory_()
            for x in self.fwd_slice:
                x.share_memory_()
            for x in self.unmasked_indices:
                for y in x:
                    y.share_memory_()

        self.num_examples = sum([len(x[0]) for x in self.unmasked_indices])
        self.type_offset = [0] + list(np.cumsum([len(x[0]) for x in self.unmasked_indices]))

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = 0
            iter_end = self.num_examples
        else:  # in a worker process
            per_worker = int(math.ceil(self.num_examples / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.num_examples)

        for idx in range(iter_start, iter_end):
            act_type_ind = np.searchsorted(self.type_offset, idx, side='right') - 1
            pos_ind = idx - self.type_offset[act_type_ind]
            i = self.unmasked_indices[act_type_ind][0][pos_ind]
            j = self.unmasked_indices[act_type_ind][1][pos_ind]

            batch_ind = self.fwd_batch[act_type_ind][i]
            graph_ind = self.batch_ind_to_graph_ind[batch_ind]
            graph = self.graphs[graph_ind]

            fwd_logits_ind = torch.tensor([batch_ind, act_type_ind, i, j], dtype=torch.long)

            tmp_action = (act_type_ind, i - self.fwd_slice[act_type_ind][batch_ind], j)
            tmp_graph_action = self.ctx.aidx_to_GraphAction(self.torch_graphs[batch_ind], tmp_action)
            if tmp_graph_action.action is GraphActionType.Stop:
                yield self.ctx.graph_to_Data(graph), \
                      torch.tensor((1.0,), dtype=torch.float32), \
                      torch.tensor(idx, dtype=torch.long), \
                      fwd_logits_ind
            else:
                tmp_graph = self.env.step(graph, tmp_graph_action)
                yield self.ctx.graph_to_Data(tmp_graph), \
                      torch.tensor((0.0,), dtype=torch.float32), \
                      torch.tensor(idx, dtype=torch.long), \
                      fwd_logits_ind

    def __len__(self):
        return self.num_examples


def get_successor_collate_fn(ctx):
    def collate_fn(batch):
        data = [x[0] for x in batch]
        terminal = torch.stack([x[1] for x in batch], dim=0)
        idx = torch.stack([x[2] for x in batch], dim=0)
        fwd_logits_ind = torch.stack([x[3] for x in batch], dim=0)

        return ctx.collate(data), terminal, idx, fwd_logits_ind
    return collate_fn


class GraphSampler:
    """A helper class to sample from GraphActionCategorical-producing models"""
    def __init__(self, ctx, env, max_len, max_nodes, rng, sample_temp=1, correct_idempotent=False):
        """
        Parameters
        ----------
        env: GraphBuildingEnv
            A graph environment.
        ctx: GraphBuildingEnvContext
            A context.
        max_len: int
            If not None, ends trajectories of more than max_len steps.
        max_nodes: int
            If not None, ends trajectories of graphs with more than max_nodes steps (illegal action).
        rng: np.random.RandomState
            rng used to take random actions
        sample_temp: float
            [Experimental] Softmax temperature used when sampling
        correct_idempotent: bool
            [Experimental] Correct for idempotent actions when counting
        """
        self.ctx = ctx
        self.env = env
        self.max_len = max_len if max_len is not None else 128
        self.max_nodes = max_nodes if max_nodes is not None else 128
        self.rng = rng
        # Experimental flags
        self.sample_temp = sample_temp
        self.sanitize_samples = True
        self.correct_idempotent = correct_idempotent

    def sample_from_model(self, model: nn.Module, n: int, cond_info: Tensor, dev: torch.device,
                          random_action_prob: float = 0.):
        """Samples a model in a minibatch

        Parameters
        ----------
        model: nn.Module
            Model whose forward() method returns GraphActionCategorical instances
        n: int
            Number of graphs to sample
        cond_info: Tensor
            Conditional information of each trajectory, shape (n, n_info)
        dev: torch.device
            Device on which data is manipulated

        Returns
        -------
        data: List[Dict]
           A list of trajectories. Each trajectory is a dict with keys
           - trajs: List[Tuple[Graph, GraphAction]], the list of states and actions
           - fwd_logprob: sum logprobs P_F
           - bck_logprob: sum logprobs P_B
           - is_valid: is the generated graph valid according to the env & ctx
        """
        # This will be returned
        data = [{'traj': [], 'reward_pred': None, 'is_valid': True} for i in range(n)]
        # Let's also keep track of trajectory statistics according to the model
        fwd_logprob: List[List[Tensor]] = [[] for i in range(n)]
        bck_logprob: List[List[Tensor]] = [[] for i in range(n)]

        graphs = [self.env.new() for i in range(n)]
        done = [False] * n

        def not_done(lst):
            return [e for i, e in enumerate(lst) if not done[i]]

        for t in range(self.max_len):
            # Construct graphs for the trajectories that aren't yet done
            torch_graphs = [self.ctx.graph_to_Data(i) for i in not_done(graphs)]
            not_done_mask = torch.tensor(done, device=dev).logical_not()
            # Forward pass to get GraphActionCategorical
            fwd_cat, log_reward_preds = model(self.ctx.collate(torch_graphs).to(dev), cond_info[not_done_mask])
            if random_action_prob > 0:
                masks = [1] * len(fwd_cat.logits) if fwd_cat.masks is None else fwd_cat.masks
                # Device which graphs in the minibatch will get their action randomized
                is_random_action = torch.tensor(
                    self.rng.uniform(size=len(torch_graphs)) < random_action_prob, device=dev).float()
                # Set the logits to some large value if they're not masked, this way the masked
                # actions have no probability of getting sampled, and there is a uniform
                # distribution over the rest
                fwd_cat.logits = [
                    # We don't multiply m by i on the right because we're assume the model forward()
                    # method already does that
                    is_random_action[b][:, None] * torch.ones_like(i) * m * 100 + i * (1 - is_random_action[b][:, None])
                    for i, m, b in zip(fwd_cat.logits, masks, fwd_cat.batch)
                ]
            if self.sample_temp != 1:
                sample_cat = copy.copy(fwd_cat)
                sample_cat.logits = [i / self.sample_temp for i in fwd_cat.logits]
                actions = sample_cat.sample()
            else:
                actions = fwd_cat.sample()
            graph_actions = [self.ctx.aidx_to_GraphAction(g, a) for g, a in zip(torch_graphs, actions)]
            log_probs = fwd_cat.log_prob(actions)
            # Step each trajectory, and accumulate statistics
            for i, j in zip(not_done(range(n)), range(n)):
                fwd_logprob[i].append(log_probs[j].unsqueeze(0))
                data[i]['traj'].append((graphs[i], graph_actions[j]))
                # Check if we're done
                if graph_actions[j].action is GraphActionType.Stop:
                    done[i] = True
                    bck_logprob[i].append(torch.tensor([1.0], device=dev).log())
                else:  # If not done, try to step the self.environment
                    gp = graphs[i]
                    try:
                        # self.env.step can raise AssertionError if the action is illegal
                        gp = self.env.step(graphs[i], graph_actions[j])
                        assert len(gp.nodes) <= self.max_nodes
                    except AssertionError:
                        done[i] = True
                        data[i]['is_valid'] = False
                        bck_logprob[i].append(torch.tensor([1.0], device=dev).log())
                        continue
                    if t == self.max_len - 1:
                        done[i] = True
                    # If no error, add to the trajectory
                    # P_B = uniform backward
                    n_back = self.env.count_backward_transitions(gp, check_idempotent=self.correct_idempotent)
                    bck_logprob[i].append(torch.tensor([1 / n_back], device=dev).log())
                    graphs[i] = gp
                if done[i] and self.sanitize_samples and not self.ctx.is_sane(graphs[i]):
                    # check if the graph is sane (e.g. RDKit can
                    # construct a molecule from it) otherwise
                    # treat the done action as illegal
                    data[i]['is_valid'] = False
            if all(done):
                break

        for i in range(n):
            # If we're not bootstrapping, we could query the reward
            # model here, but this is expensive/impractical.  Instead
            # just report forward and backward logprobs
            data[i]['fwd_logprob'] = sum(fwd_logprob[i])
            data[i]['bck_logprob'] = sum(bck_logprob[i])
            data[i]['bck_logprobs'] = torch.stack(bck_logprob[i]).reshape(-1)
            data[i]['result'] = graphs[i]
        return data

    @torch.no_grad()
    def sample_from_model_guided_joint_beta(self, model_1: nn.Module, model_2: nn.Module,
                                            graph_cls: nn.Module, n: int,
                                            cond_info_1: Tensor, cond_info_2:Tensor, dev: torch.device,
                                            random_action_prob: float = 0.,
                                            just_mixture=False, cls_y1=1, cls_y2=2, cls_max_batch_size=1000,
                                            cls_num_workers=0):
        """Samples a model in a minibatch

        Parameters
        ----------
        model: nn.Module
            Model whose forward() method returns GraphActionCategorical instances
        n: int
            Number of graphs to sample
        cond_info: Tensor
            Conditional information of each trajectory, shape (n, n_info)
        dev: torch.device
            Device on which data is manipulated

        Returns
        -------
        data: List[Dict]
           A list of trajectories. Each trajectory is a dict with keys
           - trajs: List[Tuple[Graph, GraphAction]], the list of states and actions
           - fwd_logprob: sum logprobs P_F
           - bck_logprob: sum logprobs P_B
           - is_valid: is the generated graph valid according to the env & ctx
        """
        if cls_y1 not in {1, 2}:
            raise ValueError(f'Invalid cls_y1: {cls_y1}')
        if cls_y2 not in {1, 2}:
            raise ValueError(f'Invalid cls_y2: {cls_y2}')

        # This will be returned
        data = [{'traj': [], 'reward_pred': None, 'is_valid': True} for i in range(n)]
        # Let's also keep track of trajectory statistics according to the model
        fwd_logprob: List[List[Tensor]] = [[] for i in range(n)]
        bck_logprob: List[List[Tensor]] = [[] for i in range(n)]

        graphs = [self.env.new() for i in range(n)]
        done = [False] * n

        def not_done(lst):
            return [e for i, e in enumerate(lst) if not done[i]]

        collate_fn = get_successor_collate_fn(self.ctx)

        for t in range(self.max_len):
            # Construct graphs for the trajectories that aren't yet done
            not_done_graph_inds = [i for i in range(n) if not done[i]]
            torch_graphs = [self.ctx.graph_to_Data(i) for i in not_done(graphs)]
            not_done_mask = torch.tensor(done, device=dev).logical_not()
            # Forward pass to get GraphActionCategorical
            torch_batch = self.ctx.collate(torch_graphs).to(dev)
            fwd_cat_1, log_reward_preds_1 = model_1(torch_batch, cond_info_1[not_done_mask])
            fwd_cat_2, log_reward_preds_2 = model_2(torch_batch, cond_info_2[not_done_mask])

            cur_cls_logprobs = graph_cls(torch_batch, torch_batch.x.new_zeros((torch_batch.num_graphs, 1)))
            logp_y1_eq_1_cur = torch.logsumexp(cur_cls_logprobs, dim=2)[:, 0]
            logp_y1_eq_2_cur = torch.logsumexp(cur_cls_logprobs, dim=2)[:, 1]

            # take logsoftmax of logits
            fwd_cat_1_logprob = copy.copy(fwd_cat_1)
            fwd_cat_1_logprob.logits = fwd_cat_1_logprob.logsoftmax()

            fwd_cat_2_logprob = copy.copy(fwd_cat_2)
            fwd_cat_2_logprob.logits = fwd_cat_2_logprob.logsoftmax()

            # create posterior weighted flow
            fwd_cat_mixture = copy.copy(fwd_cat_1_logprob)
            fwd_cat_mixture.logits = [
                torch.logsumexp(
                    torch.stack([logprobs_1 + logp_y1_eq_1_cur[b][:, None],
                                 logprobs_2 + logp_y1_eq_2_cur[b][:, None]], dim=0),
                    dim=0)
                for logprobs_1, logprobs_2, b in
                zip(fwd_cat_1_logprob.logits, fwd_cat_2_logprob.logits, fwd_cat_mixture.batch)
            ]

            guided_cat = copy.copy(fwd_cat_mixture)
            if not just_mixture:
                # guidance start
                successor_dataset = SuccessorGraphDataset(self.env, self.ctx,
                                                          graphs, torch_graphs,
                                                          fwd_cat_mixture.logits, fwd_cat_mixture.masks,
                                                          fwd_cat_mixture.batch, fwd_cat_mixture.slice,
                                                          not_done_graph_inds)
                num_successors = len(successor_dataset)

                tmp_cls_logprobs = torch.empty((num_successors, 2, 2), dtype=torch.float32, device=dev)
                tmp_indices = torch.empty((num_successors, 4), dtype=torch.long, device=dev)

                cls_batch_size = max(50, min(cls_max_batch_size,
                                             int(math.ceil(num_successors // max(1, cls_num_workers)))))

                loader = torch.utils.data.DataLoader(successor_dataset,
                                                     batch_size=cls_batch_size,
                                                     num_workers=cls_num_workers,
                                                     collate_fn=collate_fn,
                                                     shuffle=False, drop_last=False)

                for successor_batch in loader:
                    successor_batch_graph = successor_batch[0].to(dev)
                    successor_batch_terminal = successor_batch[1].to(dev)
                    successor_batch_idx = successor_batch[2].to(dev)
                    successor_batch_fwd_logits_ind = successor_batch[3].to(dev)

                    successor_batch_logprobs = graph_cls(successor_batch_graph, successor_batch_terminal)

                    tmp_cls_logprobs[successor_batch_idx] = successor_batch_logprobs
                    tmp_indices[successor_batch_idx] = successor_batch_fwd_logits_ind

                cur_cls_term = cur_cls_logprobs[:, cls_y1 - 1, cls_y2 - 1]
                tmp_cls_term = tmp_cls_logprobs[:, cls_y1 - 1, cls_y2 - 1]

                # tmp_indices = tensor([[batch_ind, act_type_ind, row, col], ...])
                for act_type_ind in range(len(guided_cat.logits)):
                    type_subset = tmp_indices[:, 1] == act_type_ind

                    guided_cat.logits[act_type_ind][tmp_indices[type_subset][:, 2], tmp_indices[type_subset][:, 3]] += \
                        tmp_cls_term[type_subset] - cur_cls_term[tmp_indices[type_subset][:, 0]]
                # guidance end


            masks = [1] * len(guided_cat.logits) if guided_cat.masks is None else guided_cat.masks
            if random_action_prob > 0:
                # Device which graphs in the minibatch will get their action randomized
                is_random_action = torch.tensor(
                    self.rng.uniform(size=len(torch_graphs)) < random_action_prob, device=dev).float()
                # Set the logits to some large value if they're not masked, this way the masked
                # actions have no probability of getting sampled, and there is a uniform
                # distribution over the rest
                guided_cat.logits = [
                    # We don't multiply m by i on the right because we're assume the model forward()
                    # method already does that
                    is_random_action[b][:, None] * torch.ones_like(i) * m * 100 + i * (1 - is_random_action[b][:, None])
                    for i, m, b in zip(guided_cat.logits, masks, guided_cat.batch)
                ]

            sample_cat = copy.copy(guided_cat)
            if self.sample_temp != 1:
                sample_cat.logits = [i / self.sample_temp for i in guided_cat.logits]


            actions = sample_cat.sample()
            graph_actions = [self.ctx.aidx_to_GraphAction(g, a) for g, a in zip(torch_graphs, actions)]
            log_probs = guided_cat.log_prob(actions)
            # Step each trajectory, and accumulate statistics
            for i, j in zip(not_done(range(n)), range(n)):
                fwd_logprob[i].append(log_probs[j].unsqueeze(0))
                data[i]['traj'].append((graphs[i], graph_actions[j]))
                # Check if we're done
                if graph_actions[j].action is GraphActionType.Stop:
                    done[i] = True
                    bck_logprob[i].append(torch.tensor([1.0], device=dev).log())
                else:  # If not done, try to step the self.environment
                    gp = graphs[i]
                    try:
                        # self.env.step can raise AssertionError if the action is illegal
                        gp = self.env.step(graphs[i], graph_actions[j])
                        assert len(gp.nodes) <= self.max_nodes
                    except AssertionError:
                        done[i] = True
                        data[i]['is_valid'] = False
                        bck_logprob[i].append(torch.tensor([1.0], device=dev).log())
                        continue
                    if t == self.max_len - 1:
                        done[i] = True
                    # If no error, add to the trajectory
                    # P_B = uniform backward
                    n_back = self.env.count_backward_transitions(gp, check_idempotent=self.correct_idempotent)
                    bck_logprob[i].append(torch.tensor([1 / n_back], device=dev).log())
                    graphs[i] = gp
                if done[i] and self.sanitize_samples and not self.ctx.is_sane(graphs[i]):
                    # check if the graph is sane (e.g. RDKit can
                    # construct a molecule from it) otherwise
                    # treat the done action as illegal
                    data[i]['is_valid'] = False
            if all(done):
                break

        for i in range(n):
            # If we're not bootstrapping, we could query the reward
            # model here, but this is expensive/impractical.  Instead
            # just report forward and backward logprobs
            data[i]['fwd_logprob'] = sum(fwd_logprob[i])
            data[i]['bck_logprob'] = sum(bck_logprob[i])
            data[i]['bck_logprobs'] = torch.stack(bck_logprob[i]).reshape(-1)
            data[i]['result'] = graphs[i]
        return data

    @torch.no_grad()
    def sample_from_model_guided_3joint_beta(self, model_1: nn.Module, model_2: nn.Module, model_3: nn.Module,
                                             graph_cls: nn.Module, n: int,
                                             cond_info_1: Tensor, cond_info_2: Tensor, cond_info_3: Tensor,
                                             dev: torch.device,
                                             random_action_prob: float = 0.,
                                             just_mixture=False, cls_y1=1, cls_y2=2, cls_y3=3,
                                             cls_max_batch_size=1000, cls_num_workers=0):
        """Samples a model in a minibatch

        Parameters
        ----------
        model: nn.Module
            Model whose forward() method returns GraphActionCategorical instances
        n: int
            Number of graphs to sample
        cond_info: Tensor
            Conditional information of each trajectory, shape (n, n_info)
        dev: torch.device
            Device on which data is manipulated

        Returns
        -------
        data: List[Dict]
           A list of trajectories. Each trajectory is a dict with keys
           - trajs: List[Tuple[Graph, GraphAction]], the list of states and actions
           - fwd_logprob: sum logprobs P_F
           - bck_logprob: sum logprobs P_B
           - is_valid: is the generated graph valid according to the env & ctx
        """
        if cls_y1 not in {1, 2, 3, None}:
            raise ValueError(f'Invalid cls_y1: {cls_y1}')
        if cls_y2 not in {1, 2, 3, None}:
            raise ValueError(f'Invalid cls_y2: {cls_y2}')
        if cls_y3 not in {1, 2, 3, None}:
            raise ValueError(f'Invalid cls_y3: {cls_y3}')

        # This will be returned
        data = [{'traj': [], 'reward_pred': None, 'is_valid': True} for i in range(n)]
        # Let's also keep track of trajectory statistics according to the model
        fwd_logprob: List[List[Tensor]] = [[] for i in range(n)]
        bck_logprob: List[List[Tensor]] = [[] for i in range(n)]

        graphs = [self.env.new() for i in range(n)]
        done = [False] * n

        def not_done(lst):
            return [e for i, e in enumerate(lst) if not done[i]]

        collate_fn = get_successor_collate_fn(self.ctx)

        for t in range(self.max_len):
            # Construct graphs for the trajectories that aren't yet done
            not_done_graph_inds = [i for i in range(n) if not done[i]]
            torch_graphs = [self.ctx.graph_to_Data(i) for i in not_done(graphs)]
            not_done_mask = torch.tensor(done, device=dev).logical_not()
            # Forward pass to get GraphActionCategorical
            torch_batch = self.ctx.collate(torch_graphs).to(dev)
            fwd_cat_1, log_reward_preds_1 = model_1(torch_batch, cond_info_1[not_done_mask])
            fwd_cat_2, log_reward_preds_2 = model_2(torch_batch, cond_info_2[not_done_mask])
            fwd_cat_3, log_reward_preds_3 = model_3(torch_batch, cond_info_3[not_done_mask])

            cur_cls_logprobs = graph_cls(torch_batch, torch_batch.x.new_zeros((torch_batch.num_graphs, 1)))
            logp_y1_eq_1_cur = torch.logsumexp(cur_cls_logprobs, dim=(2, 3))[:, 0]
            logp_y1_eq_2_cur = torch.logsumexp(cur_cls_logprobs, dim=(2, 3))[:, 1]
            logp_y1_eq_3_cur = torch.logsumexp(cur_cls_logprobs, dim=(2, 3))[:, 2]

            # take logsoftmax of logits
            fwd_cat_1_logprob = copy.copy(fwd_cat_1)
            fwd_cat_1_logprob.logits = fwd_cat_1_logprob.logsoftmax()

            fwd_cat_2_logprob = copy.copy(fwd_cat_2)
            fwd_cat_2_logprob.logits = fwd_cat_2_logprob.logsoftmax()

            fwd_cat_3_logprob = copy.copy(fwd_cat_3)
            fwd_cat_3_logprob.logits = fwd_cat_3_logprob.logsoftmax()

            # create posterior weighted flow
            fwd_cat_mixture = copy.copy(fwd_cat_1_logprob)
            fwd_cat_mixture.logits = [
                torch.logsumexp(
                    torch.stack([logprobs_1 + logp_y1_eq_1_cur[b][:, None],
                                 logprobs_2 + logp_y1_eq_2_cur[b][:, None],
                                 logprobs_3 + logp_y1_eq_3_cur[b][:, None]], dim=0),
                    dim=0)
                for logprobs_1, logprobs_2, logprobs_3, b in
                zip(fwd_cat_1_logprob.logits, fwd_cat_2_logprob.logits, fwd_cat_3_logprob.logits, fwd_cat_mixture.batch)
            ]

            guided_cat = copy.copy(fwd_cat_mixture)
            if not just_mixture:
                # guidance start
                successor_dataset = SuccessorGraphDataset(self.env, self.ctx,
                                                          graphs, torch_graphs,
                                                          fwd_cat_mixture.logits, fwd_cat_mixture.masks,
                                                          fwd_cat_mixture.batch, fwd_cat_mixture.slice,
                                                          not_done_graph_inds)
                # share_memory=cls_num_workers > 0)
                num_successors = len(successor_dataset)

                tmp_cls_logprobs = torch.empty((num_successors, 3, 3, 3), dtype=torch.float32, device=dev)
                tmp_indices = torch.empty((num_successors, 4), dtype=torch.long, device=dev)

                cls_batch_size = max(50, min(cls_max_batch_size,
                                             int(math.ceil(num_successors // max(1, cls_num_workers)))))

                loader = torch.utils.data.DataLoader(successor_dataset,
                                                     batch_size=cls_batch_size,
                                                     num_workers=cls_num_workers,
                                                     collate_fn=collate_fn,
                                                     shuffle=False, drop_last=False)

                for successor_batch in loader:
                    successor_batch_graph = successor_batch[0].to(dev)
                    successor_batch_terminal = successor_batch[1].to(dev)
                    successor_batch_idx = successor_batch[2].to(dev)
                    successor_batch_fwd_logits_ind = successor_batch[3].to(dev)

                    successor_batch_logprobs = graph_cls(successor_batch_graph, successor_batch_terminal)

                    tmp_cls_logprobs[successor_batch_idx] = successor_batch_logprobs
                    tmp_indices[successor_batch_idx] = successor_batch_fwd_logits_ind


                # cur_cls_term = cur_cls_logprobs[:, cls_y1 - 1, cls_y2 - 1, cls_y3 - 1]
                # tmp_cls_term = tmp_cls_logprobs[:, cls_y1 - 1, cls_y2 - 1, cls_y3 - 1]

                cur_cls_term = extract_logprobs(cur_cls_logprobs, cls_y1, cls_y2, cls_y3)
                tmp_cls_term = extract_logprobs(tmp_cls_logprobs, cls_y1, cls_y2, cls_y3)


                for act_type_ind in range(len(guided_cat.logits)):
                    type_subset = tmp_indices[:, 1] == act_type_ind

                    guided_cat.logits[act_type_ind][tmp_indices[type_subset][:, 2], tmp_indices[type_subset][:, 3]] += \
                        tmp_cls_term[type_subset] - cur_cls_term[tmp_indices[type_subset][:, 0]]
                # guidance end


            masks = [1] * len(guided_cat.logits) if guided_cat.masks is None else guided_cat.masks
            if random_action_prob > 0:
                # Device which graphs in the minibatch will get their action randomized
                is_random_action = torch.tensor(
                    self.rng.uniform(size=len(torch_graphs)) < random_action_prob, device=dev).float()
                # Set the logits to some large value if they're not masked, this way the masked
                # actions have no probability of getting sampled, and there is a uniform
                # distribution over the rest
                guided_cat.logits = [
                    # We don't multiply m by i on the right because we're assume the model forward()
                    # method already does that
                    is_random_action[b][:, None] * torch.ones_like(i) * m * 100 + i * (1 - is_random_action[b][:, None])
                    for i, m, b in zip(guided_cat.logits, masks, guided_cat.batch)
                ]

            sample_cat = copy.copy(guided_cat)
            if self.sample_temp != 1:
                sample_cat.logits = [i / self.sample_temp for i in guided_cat.logits]


            actions = sample_cat.sample()
            graph_actions = [self.ctx.aidx_to_GraphAction(g, a) for g, a in zip(torch_graphs, actions)]
            log_probs = guided_cat.log_prob(actions)
            # Step each trajectory, and accumulate statistics
            for i, j in zip(not_done(range(n)), range(n)):
                fwd_logprob[i].append(log_probs[j].unsqueeze(0))
                data[i]['traj'].append((graphs[i], graph_actions[j]))
                # Check if we're done
                if graph_actions[j].action is GraphActionType.Stop:
                    done[i] = True
                    bck_logprob[i].append(torch.tensor([1.0], device=dev).log())
                else:  # If not done, try to step the self.environment
                    gp = graphs[i]
                    try:
                        # self.env.step can raise AssertionError if the action is illegal
                        gp = self.env.step(graphs[i], graph_actions[j])
                        assert len(gp.nodes) <= self.max_nodes
                    except AssertionError:
                        done[i] = True
                        data[i]['is_valid'] = False
                        bck_logprob[i].append(torch.tensor([1.0], device=dev).log())
                        continue
                    if t == self.max_len - 1:
                        done[i] = True
                    # If no error, add to the trajectory
                    # P_B = uniform backward
                    n_back = self.env.count_backward_transitions(gp, check_idempotent=self.correct_idempotent)
                    bck_logprob[i].append(torch.tensor([1 / n_back], device=dev).log())
                    graphs[i] = gp
                if done[i] and self.sanitize_samples and not self.ctx.is_sane(graphs[i]):
                    # check if the graph is sane (e.g. RDKit can
                    # construct a molecule from it) otherwise
                    # treat the done action as illegal
                    data[i]['is_valid'] = False
            if all(done):
                break

        for i in range(n):
            # If we're not bootstrapping, we could query the reward
            # model here, but this is expensive/impractical.  Instead
            # just report forward and backward logprobs
            data[i]['fwd_logprob'] = sum(fwd_logprob[i])
            data[i]['bck_logprob'] = sum(bck_logprob[i])
            data[i]['bck_logprobs'] = torch.stack(bck_logprob[i]).reshape(-1)
            data[i]['result'] = graphs[i]
        return data
