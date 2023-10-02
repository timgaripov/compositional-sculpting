import ast
import copy
import math
import time
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import numpy as np
import scipy.stats as stats
from rdkit import RDLogger
from rdkit.Chem.rdchem import Mol as RDMol
from rdkit.Chem import Descriptors
from rdkit.Chem import QED
from torch import Tensor
import torch.nn as nn
from torch.distributions.dirichlet import Dirichlet
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.utils.tensorboard
import torch_geometric.data as gd

from params_proto import ParamsProto, PrefixProto

from gflownet.algo.trajectory_balance import TrajectoryBalance
from gflownet.data.sampling_iterator import SamplingIterator
from gflownet.envs.frag_mol_env import FragMolBuildingEnvContext
from gflownet.envs.graph_building_env import GraphBuildingEnv, GraphActionCategorical
from gflownet.models import bengio2021flow
from gflownet.models.graph_transformer import GraphTransformerGFN
from gflownet.utils import metrics
from gflownet.utils import sascore
from gflownet.utils.multiobjective_hooks import MultiObjectiveStatsHook
from gflownet.utils.multiobjective_hooks import TopKHook
from gflownet.utils.multiprocessing_proxy import wrap_model_mp
from gflownet.utils.transforms import thermometer


class Args(ParamsProto, prefix='gflownet'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    num_emb = 128
    num_layers = 6
    weight_decay = 1e-8
    momentum = 0.9
    adam_eps = 1e-8
    clip_grad_type = 'norm'
    clip_grad_param = 10
    valid_random_action_prob = 0.

    # SEHMOOFragTrainer
    use_fixed_weight = False
    valid_sample_cond_info = False

    # seh_frag_moo.py main()
    seed = 100
    global_batch_size = 64
    num_training_steps = 20_000
    validate_every = 125
    algo = 'TB'
    objectives = ['seh']
    learning_rate = 0.0005
    Z_learning_rate = 0.0005
    lr_decay = 20000
    Z_lr_decay = 50000
    sampling_tau = 0.95
    random_action_prob = 0.1
    num_data_loader_workers = 8
    temperature_sample_dist = 'uniform'
    temperature_dist_params = [0.0, 96.0]
    num_thermometer_dim = 32
    preference_type = 'seeded_single'
    n_valid_prefs = 15
    n_valid_repeats_per_pref = 128

    class trajectory_balance(PrefixProto, cli=False):
        illegal_action_logreward = -75
        reward_loss_multiplier = 1
        bootstrap_own_reward = False
        epsilon = None

        do_subtb = False
        correct_idempotent = False
        subtb_max_len = None


# This type represents an unprocessed list of reward signals/conditioning information
FlatRewards = NewType('FlatRewards', Tensor)  # type: ignore

# This type represents the outcome for a multi-objective task of
# converting FlatRewards to a scalar, e.g. (sum R_i omega_i) ** beta
RewardScalar = NewType('RewardScalar', Tensor)  # type: ignore


def cycle(it):
    while True:
        for i in it:
            yield i


class RepeatedPreferenceDataset:
    def __init__(self, preferences, repeat):
        self.prefs = preferences
        self.repeat = repeat

    def __len__(self):
        return len(self.prefs) * self.repeat

    def __getitem__(self, idx):
        assert 0 <= idx < len(self)
        return torch.tensor(self.prefs[int(idx // self.repeat)])


class SEHSOOTask(object):
    """Sets up a sinlge objective task where the rewards is one of (functions of):
    - the the binding energy of a molecule to Soluble Epoxide Hydrolases.
    - its QED
    - its synthetic accessibility
    - its molecular weight

    The proxy is pretrained, and obtained from the original GFlowNet paper, see `gflownet.models.bengio2021flow`.
    """
    def __init__(self, objectives: List[str], dataset: Dataset, temperature_sample_dist: str,
                 temperature_parameters: Tuple[float, float], num_thermometer_dim: int, rng: np.random.Generator = None,
                 wrap_model: Callable[[nn.Module], nn.Module] = None):
        self._wrap_model = wrap_model
        self.rng = rng
        self.models = self._load_task_models()
        self.objectives = objectives
        self.dataset = dataset
        self.temperature_sample_dist = temperature_sample_dist
        self.temperature_dist_params = temperature_parameters
        self.num_thermometer_dim = num_thermometer_dim
        self.seeded_preference = None
        self.experimental_dirichlet = False

        objectives_set = {
            'seh', 'qed', 'sa', 'mw',
        }

        assert set(objectives) <= objectives_set and len(objectives) == 1


    def flat_reward_transform(self, y: Union[float, Tensor]) -> FlatRewards:
        return FlatRewards(torch.as_tensor(y))

    def inverse_flat_reward_transform(self, rp):
        return rp

    def _load_task_models(self):
        model = bengio2021flow.load_original_model()
        model, self.device = self._wrap_model(model)
        return {'seh': model}

    def sample_conditional_information(self, n: int) -> Dict[str, Tensor]:
        # SEHTask sample_conditional_information()
        beta = None
        if self.temperature_sample_dist == 'constant':
            assert type(self.temperature_dist_params) is float
            beta = np.array(self.temperature_dist_params).repeat(n).astype(np.float32)
            beta_enc = torch.zeros((n, self.num_thermometer_dim))
        else:
            if self.temperature_sample_dist == 'gamma':
                loc, scale = self.temperature_dist_params
                beta = self.rng.gamma(loc, scale, n).astype(np.float32)
                upper_bound = stats.gamma.ppf(0.95, loc, scale=scale)
            elif self.temperature_sample_dist == 'uniform':
                beta = self.rng.uniform(*self.temperature_dist_params, n).astype(np.float32)
                upper_bound = self.temperature_dist_params[1]
            elif self.temperature_sample_dist == 'loguniform':
                low, high = np.log(self.temperature_dist_params)
                beta = np.exp(self.rng.uniform(low, high, n).astype(np.float32))
                upper_bound = self.temperature_dist_params[1]
            elif self.temperature_sample_dist == 'beta':
                beta = self.rng.beta(*self.temperature_dist_params, n).astype(np.float32)
                upper_bound = 1
            beta_enc = thermometer(torch.tensor(beta), self.num_thermometer_dim, 0, upper_bound)

        assert len(beta.shape) == 1, f"beta should be a 1D array, got {beta.shape}"
        cond_info = {'beta': beta, 'encoding': beta_enc}

        # END SEHTask sample_conditional_information()

        return cond_info

    def encode_conditional_information(self, preferences: torch.TensorType) -> Dict[str, Tensor]:
        if self.temperature_sample_dist == 'constant':
            beta = torch.ones(len(preferences)) * self.temperature_dist_params
            beta_enc = torch.zeros((len(preferences), self.num_thermometer_dim))
        else:
            beta = torch.ones(len(preferences)) * self.temperature_dist_params[-1]
            beta_enc = torch.ones((len(preferences), self.num_thermometer_dim))

        assert len(beta.shape) == 1, f"beta should be of shape (Batch,), got: {beta.shape}"
        # ignore preferences
        encoding = beta_enc
        return {'beta': beta, 'encoding': encoding.float()}


    def encode_conditional_information_custom_beta(self,
                                                   beta: Optional[torch.TensorType],
                                                   batch_size: int) -> Dict[str, Tensor]:

        assert self.temperature_sample_dist != 'constant'

        upper_bound = None
        if self.temperature_sample_dist == 'gamma':
            loc, scale = self.temperature_dist_params
            upper_bound = stats.gamma.ppf(0.95, loc, scale=scale)
        elif self.temperature_sample_dist == 'uniform':
            upper_bound = self.temperature_dist_params[1]
        elif self.temperature_sample_dist == 'loguniform':
            upper_bound = self.temperature_dist_params[1]
        elif self.temperature_sample_dist == 'beta':
            upper_bound = 1
        assert upper_bound is not None

        if beta is None:
            beta = torch.full((batch_size,), upper_bound)
        assert len(beta.shape) == 1, f"beta should be of shape (Batch,), got: {beta.shape}"

        beta_enc = thermometer(beta, self.num_thermometer_dim, 0, upper_bound)
        encoding = beta_enc

        return {'beta': beta, 'encoding': encoding.float()}

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
        if isinstance(flat_reward, list):
            if isinstance(flat_reward[0], Tensor):
                flat_reward = torch.stack(flat_reward)
            else:
                flat_reward = torch.tensor(flat_reward)
        scalar_logreward = flat_reward.squeeze().clamp(min=1e-30).log()
        assert len(scalar_logreward.shape) == len(cond_info['beta'].shape), \
            f"dangerous shape mismatch: {scalar_logreward.shape} vs {cond_info['beta'].shape}"
        return RewardScalar(scalar_logreward * cond_info['beta'])

    def compute_flat_rewards(self, mols: List[RDMol]) -> Tuple[FlatRewards, Tensor]:
        graphs = [bengio2021flow.mol2graph(i) for i in mols]
        is_valid = torch.tensor([i is not None for i in graphs]).bool()
        if not is_valid.any():
            return FlatRewards(torch.zeros((0, len(self.objectives)))), is_valid
        else:
            flat_rewards = []
            if 'seh' in self.objectives:
                batch = gd.Batch.from_data_list([i for i in graphs if i is not None])
                batch.to(self.device)
                seh_preds = self.models['seh'](batch).reshape((-1,)).clip(1e-4, 100).data.cpu() / 8
                seh_preds[seh_preds.isnan()] = 0
                flat_rewards.append(seh_preds)

            def safe(f, x, default):
                try:
                    return f(x)
                except Exception:
                    return default

            if "qed" in self.objectives:
                qeds = torch.tensor([safe(QED.qed, i, 0) for i, v in zip(mols, is_valid) if v.item()])
                flat_rewards.append(qeds)

            if "sa" in self.objectives:
                sas = torch.tensor([safe(sascore.calculateScore, i, 10) for i, v in zip(mols, is_valid) if v.item()])
                sas = (10 - sas) / 9  # Turn into a [0-1] reward
                flat_rewards.append(sas)

            if "mw" in self.objectives:
                molwts = torch.tensor([safe(Descriptors.MolWt, i, 1000) for i, v in zip(mols, is_valid) if v.item()])
                molwts = ((300 - molwts) / 700 + 1).clip(0, 1)  # 1 until 300 then linear decay to 0 until 1000
                flat_rewards.append(molwts)

            flat_rewards = torch.stack(flat_rewards, dim=1)
            return FlatRewards(flat_rewards), is_valid


# Adapt SEHMOOTask without subclassing
class SEHMOOTask(object):
    """Sets up a multiobjective task where the rewards are (functions of):
    - the the binding energy of a molecule to Soluble Epoxide Hydrolases.
    - its QED
    - its synthetic accessibility
    - its molecular weight

    The proxy is pretrained, and obtained from the original GFlowNet paper, see `gflownet.models.bengio2021flow`.
    """
    def __init__(self, objectives: List[str], dataset: Dataset, temperature_sample_dist: str,
                 temperature_parameters: Tuple[float, float], num_thermometer_dim: int, rng: np.random.Generator = None,
                 wrap_model: Callable[[nn.Module], nn.Module] = None):
        self._wrap_model = wrap_model
        self.rng = rng
        self.models = self._load_task_models()
        self.objectives = objectives
        self.dataset = dataset
        self.temperature_sample_dist = temperature_sample_dist
        self.temperature_dist_params = temperature_parameters
        self.num_thermometer_dim = num_thermometer_dim
        self.seeded_preference = None
        self.experimental_dirichlet = False

        objectives_set = {
            'seh', 'qed', 'sa', 'mw',
        }

        assert set(objectives) <= objectives_set


    def flat_reward_transform(self, y: Union[float, Tensor]) -> FlatRewards:
        return FlatRewards(torch.as_tensor(y))

    def inverse_flat_reward_transform(self, rp):
        return rp

    def _load_task_models(self):
        model = bengio2021flow.load_original_model()
        model, self.device = self._wrap_model(model)
        return {'seh': model}

    def sample_conditional_information(self, n: int) -> Dict[str, Tensor]:
        # SEHTask sample_conditional_information()
        beta = None
        if self.temperature_sample_dist == 'constant':
            assert type(self.temperature_dist_params) is float
            beta = np.array(self.temperature_dist_params).repeat(n).astype(np.float32)
            beta_enc = torch.zeros((n, self.num_thermometer_dim))
        else:
            if self.temperature_sample_dist == 'gamma':
                loc, scale = self.temperature_dist_params
                beta = self.rng.gamma(loc, scale, n).astype(np.float32)
                upper_bound = stats.gamma.ppf(0.95, loc, scale=scale)
            elif self.temperature_sample_dist == 'uniform':
                beta = self.rng.uniform(*self.temperature_dist_params, n).astype(np.float32)
                upper_bound = self.temperature_dist_params[1]
            elif self.temperature_sample_dist == 'loguniform':
                low, high = np.log(self.temperature_dist_params)
                beta = np.exp(self.rng.uniform(low, high, n).astype(np.float32))
                upper_bound = self.temperature_dist_params[1]
            elif self.temperature_sample_dist == 'beta':
                beta = self.rng.beta(*self.temperature_dist_params, n).astype(np.float32)
                upper_bound = 1
            beta_enc = thermometer(torch.tensor(beta), self.num_thermometer_dim, 0, upper_bound)

        assert len(beta.shape) == 1, f"beta should be a 1D array, got {beta.shape}"
        cond_info = {'beta': beta, 'encoding': beta_enc}

        # END SEHTask sample_conditional_information()

        if self.seeded_preference is not None:
            preferences = torch.tensor([self.seeded_preference] * n).float()
        elif self.experimental_dirichlet:
            a = np.random.dirichlet([1] * len(self.objectives), n)
            b = np.random.exponential(1, n)[:, None]
            preferences = Dirichlet(torch.tensor(a * b)).sample([1])[0].float()
        else:
            m = Dirichlet(torch.FloatTensor([1.] * len(self.objectives)))
            preferences = m.sample([n])

        cond_info['encoding'] = torch.cat([cond_info['encoding'], preferences], 1)
        cond_info['preferences'] = preferences
        return cond_info

    def encode_conditional_information(self, preferences: torch.TensorType) -> Dict[str, Tensor]:
        if self.temperature_sample_dist == 'constant':
            beta = torch.ones(len(preferences)) * self.temperature_dist_params
            beta_enc = torch.zeros((len(preferences), self.num_thermometer_dim))
        else:
            beta = torch.ones(len(preferences)) * self.temperature_dist_params[-1]
            beta_enc = torch.ones((len(preferences), self.num_thermometer_dim))

        assert len(beta.shape) == 1, f"beta should be of shape (Batch,), got: {beta.shape}"
        encoding = torch.cat([beta_enc, preferences], 1)
        return {'beta': beta, 'encoding': encoding.float(), 'preferences': preferences.float()}

    def encode_conditional_information_custom_beta(self,
                                                   beta: Optional[torch.TensorType],
                                                   preferences: torch.TensorType) -> Dict[str, Tensor]:


        assert self.temperature_sample_dist != 'constant'

        upper_bound = None
        if self.temperature_sample_dist == 'gamma':
            loc, scale = self.temperature_dist_params
            upper_bound = stats.gamma.ppf(0.95, loc, scale=scale)
        elif self.temperature_sample_dist == 'uniform':
            upper_bound = self.temperature_dist_params[1]
        elif self.temperature_sample_dist == 'loguniform':
            upper_bound = self.temperature_dist_params[1]
        elif self.temperature_sample_dist == 'beta':
            upper_bound = 1
        assert upper_bound is not None

        if beta is None:
            beta = torch.full((len(preferences),), upper_bound)
        assert len(beta.shape) == 1, f"beta should be of shape (Batch,), got: {beta.shape}"

        beta_enc = thermometer(beta, self.num_thermometer_dim, 0, upper_bound)
        encoding = torch.cat([beta_enc, preferences], 1)

        return {'beta': beta, 'encoding': encoding.float(), 'preferences': preferences.float()}

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
        if isinstance(flat_reward, list):
            if isinstance(flat_reward[0], Tensor):
                flat_reward = torch.stack(flat_reward)
            else:
                flat_reward = torch.tensor(flat_reward)
        scalar_logreward = (flat_reward * cond_info['preferences']).sum(1).clamp(min=1e-30).log()
        assert len(scalar_logreward.shape) == len(cond_info['beta'].shape), \
            f"dangerous shape mismatch: {scalar_logreward.shape} vs {cond_info['beta'].shape}"
        return RewardScalar(scalar_logreward * cond_info['beta'])

    def compute_flat_rewards(self, mols: List[RDMol]) -> Tuple[FlatRewards, Tensor]:
        graphs = [bengio2021flow.mol2graph(i) for i in mols]
        is_valid = torch.tensor([i is not None for i in graphs]).bool()
        if not is_valid.any():
            return FlatRewards(torch.zeros((0, len(self.objectives)))), is_valid

        else:
            flat_rewards = []
            if 'seh' in self.objectives:
                batch = gd.Batch.from_data_list([i for i in graphs if i is not None])
                batch.to(self.device)
                seh_preds = self.models['seh'](batch).reshape((-1,)).clip(1e-4, 100).data.cpu() / 8
                seh_preds[seh_preds.isnan()] = 0
                flat_rewards.append(seh_preds)

            def safe(f, x, default):
                try:
                    return f(x)
                except Exception:
                    return default

            if "qed" in self.objectives:
                qeds = torch.tensor([safe(QED.qed, i, 0) for i, v in zip(mols, is_valid) if v.item()])
                flat_rewards.append(qeds)

            if "sa" in self.objectives:
                sas = torch.tensor([safe(sascore.calculateScore, i, 10) for i, v in zip(mols, is_valid) if v.item()])
                sas = (10 - sas) / 9  # Turn into a [0-1] reward
                flat_rewards.append(sas)

            if "mw" in self.objectives:
                molwts = torch.tensor([safe(Descriptors.MolWt, i, 1000) for i, v in zip(mols, is_valid) if v.item()])
                molwts = ((300 - molwts) / 700 + 1).clip(0, 1)  # 1 until 300 then linear decay to 0 until 1000
                flat_rewards.append(molwts)


            flat_rewards = torch.stack(flat_rewards, dim=1)
            return FlatRewards(flat_rewards), is_valid

    def flat_reward_names(self) -> List[str]:
        names = []
        name_order = ['seh', 'qed', 'sa', 'mw']
        for name in name_order:
            if name in self.objectives:
                names.append(name)
        return names


# rewriting GFNTrainer, so that it implements SEHMOOFragTrainer without subclassing
class Trainer:
    # replace hps with Args
    def __init__(self, Args: Args, device: torch.device, setup_ml_logger=True):
        """A GFlowNet trainer. Contains the main training loop in `run` and should be subclassed.

        Parameters
        ----------
        device: torch.device
            The torch device of the main worker.
        """

        self.Args = Args
        self.device = device
        # The number of processes spawned to sample object and do CPU work
        self.num_workers: int = self.Args.num_data_loader_workers

        self.verbose = False
        # These hooks allow us to compute extra quantities when sampling data
        self.sampling_hooks: List[Callable] = []
        self.valid_sampling_hooks: List[Callable] = []
        # Will check if parameters are finite at every iteration (can be costly)
        self._validate_parameters = False

        self.setup(setup_ml_logger=setup_ml_logger)


    def setup_env_context(self):
        # if single objective, no preference encoding
        preference_enc_dim = 0 if len(self.Args.objectives) == 1 else len(self.Args.objectives)
        self.ctx = FragMolBuildingEnvContext(max_frags=9,
                                             num_cond_dim=self.Args.num_thermometer_dim + preference_enc_dim)

    def setup_algo(self):
        if self.Args.algo == 'TB':
            self.algo = TrajectoryBalance(self.env, self.ctx, self.rng, self.Args.trajectory_balance, max_nodes=9)
        else:
            raise NotImplementedError(f'{self.Args.algo} is not implemented')

    def setup_task(self):
        if len(self.Args.objectives) == 1:
            self.task = SEHSOOTask(objectives=self.Args.objectives, dataset=self.training_data,
                                   temperature_sample_dist=self.Args.temperature_sample_dist,
                                   temperature_parameters=self.Args.temperature_dist_params,
                                   num_thermometer_dim=self.Args.num_thermometer_dim, wrap_model=self._wrap_model_mp)
        else:
            self.task = SEHMOOTask(objectives=self.Args.objectives, dataset=self.training_data,
                                   temperature_sample_dist=self.Args.temperature_sample_dist,
                                   temperature_parameters=self.Args.temperature_dist_params,
                                   num_thermometer_dim=self.Args.num_thermometer_dim, wrap_model=self._wrap_model_mp)

    def setup_model(self):
        if self.Args.algo == 'TB':
            model = GraphTransformerGFN(self.ctx, num_emb=self.Args.num_emb, num_layers=self.Args.num_layers)
        else:
            raise NotImplementedError(f'{self.Args.algo} is not implemented')

        if self.Args.algo in ['A2C', 'MOQL']:
            model.do_mask = False
        self.model = model

    def setup(self, setup_ml_logger=True):
        # SEHFragTrainer.setup()
        if setup_ml_logger:
            from ml_logger import logger

            logger.log_params(Args=vars(self.Args))

            logger.log_text("""
            charts:
            - yKey: train/loss/mean
              xKey: step
            - yKey: train/logZ/mean
              xKey: step
            - yKey: train/lifetime_hv0/max
              xKey: step
            - yKey: valid/loss/mean
              xKey: step
            - yKey: valid/logZ/mean
              xKey: step
            - yKey: valid_end/topk_rewards_avg/mean
              xKey: step
            - yKey: steps_per_sec/mean
              xKey: step
            """, ".charts.yml", dedent=True)

        args = self.Args

        # preprocess args
        eps = args.trajectory_balance.epsilon
        args.trajectory_balance.epsilon = ast.literal_eval(eps) if isinstance(eps, str) else eps

        RDLogger.DisableLog('rdApp.*')
        self.rng = np.random.default_rng(142857)
        self.env = GraphBuildingEnv()
        self.training_data = []
        self.test_data = []
        self.offline_ratio = 0
        self.valid_offline_ratio = 0
        self.setup_env_context()
        self.setup_algo()
        self.setup_task()
        self.setup_model()

        # Separate Z parameters from non-Z to allow for LR decay on the former
        Z_params = list(self.model.logZ.parameters())
        non_Z_params = [i for i in self.model.parameters() if all(id(i) != id(j) for j in Z_params)]
        self.opt = torch.optim.Adam(non_Z_params, args.learning_rate, (args.momentum, 0.999),
                                    weight_decay=args.weight_decay, eps=args.adam_eps)
        self.opt_Z = torch.optim.Adam(Z_params, args.Z_learning_rate, (0.9, 0.999))
        self.lr_sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda steps: 2**(-steps / args.lr_decay))
        self.lr_sched_Z = torch.optim.lr_scheduler.LambdaLR(self.opt_Z, lambda steps: 2**(-steps / args.Z_lr_decay))

        self.sampling_tau = args.sampling_tau
        if self.sampling_tau > 0:
            self.sampling_model = copy.deepcopy(self.model)
        else:
            self.sampling_model = self.model

        self.mb_size = args.global_batch_size
        self.clip_grad_param = args.clip_grad_param
        self.clip_grad_callback = {
            'value': (lambda params: torch.nn.utils.clip_grad_value_(params, self.clip_grad_param)),
            'norm': (lambda params: torch.nn.utils.clip_grad_norm_(params, self.clip_grad_param)),
            'none': (lambda x: None)
        }[args.clip_grad_type]
        # END SEHFragTrainer.setup()

        self.sampling_hooks.append(MultiObjectiveStatsHook(256))

        n_obj = len(args.objectives)
        if args.preference_type == 'dirichlet':
            valid_preferences = metrics.generate_simplex(n_obj, n_per_dim=math.ceil(args.n_valid_prefs / n_obj))
        elif args.preference_type == 'seeded_single':
            seeded_prefs = np.random.default_rng(142857 + int(args.seed)).dirichlet([1] * n_obj,
                                                                                    args.n_valid_prefs)
            valid_preferences = seeded_prefs[0].reshape((1, n_obj))
            self.task.seeded_preference = valid_preferences[0]
        elif args.preference_type == 'seeded_many':
            valid_preferences = np.random.default_rng(142857 + int(args.seed)).dirichlet(
                [1] * n_obj, args.n_valid_prefs)

        self._top_k_hook = TopKHook(10, args.n_valid_repeats_per_pref, len(valid_preferences))
        self.test_data = RepeatedPreferenceDataset(valid_preferences, args.n_valid_repeats_per_pref)
        self.valid_sampling_hooks.append(self._top_k_hook)

        self.algo.task = self.task


    def step(self, loss: Tensor):
        loss.backward()
        for i in self.model.parameters():
            self.clip_grad_callback(i)
        self.opt.step()
        self.opt.zero_grad()
        self.opt_Z.step()
        self.opt_Z.zero_grad()
        self.lr_sched.step()
        self.lr_sched_Z.step()
        if self.sampling_tau > 0:
            for a, b in zip(self.model.parameters(), self.sampling_model.parameters()):
                b.data.mul_(self.sampling_tau).add_(a.data * (1 - self.sampling_tau))

    def _wrap_model_mp(self, model):
        """Wraps a nn.Module instance so that it can be shared to `DataLoader` workers.  """
        model.to(self.device)
        if self.num_workers > 0:
            placeholder = wrap_model_mp(model, self.num_workers, cast_types=(gd.Batch, GraphActionCategorical))
            return placeholder, torch.device('cpu')
        return model, self.device

    def build_callbacks(self):
        # We use this class-based setup to be compatible with the DeterminedAI API, but no direct
        # dependency is required.
        parent = self

        class TopKMetricCB:
            def on_validation_end(self, metrics: Dict[str, Any]):
                top_k = parent._top_k_hook.finalize()
                for i in range(len(top_k)):
                    metrics[f'topk_rewards_{i}'] = top_k[i]
                metrics[f'topk_rewards_avg'] = np.mean(top_k)
                from ml_logger import logger
                logger.print('validation end', metrics)

        return {'topk': TopKMetricCB()}

    def build_training_data_loader(self) -> DataLoader:
        model, dev = self._wrap_model_mp(self.sampling_model)
        # TODO: figure out where to save logs
        iterator = SamplingIterator(self.training_data, model, self.mb_size, self.ctx, self.algo, self.task, dev,
                                    ratio=self.offline_ratio,
                                    logger_dir='train',
                                    random_action_prob=self.Args.random_action_prob)
        for hook in self.sampling_hooks:
            iterator.add_log_hook(hook)
        return torch.utils.data.DataLoader(iterator, batch_size=None, num_workers=self.num_workers,
                                           persistent_workers=self.num_workers > 0)

    def build_validation_data_loader(self) -> DataLoader:
        model, dev = self._wrap_model_mp(self.model)
        # TODO: figure out where to save logs
        iterator = SamplingIterator(self.test_data, model, self.mb_size, self.ctx, self.algo, self.task, dev,
                                    ratio=self.valid_offline_ratio,
                                    logger_dir='valid',
                                    sample_cond_info=self.Args.valid_sample_cond_info, stream=False,
                                    random_action_prob=self.Args.valid_random_action_prob)
        for hook in self.valid_sampling_hooks:
            iterator.add_log_hook(hook)
        return torch.utils.data.DataLoader(iterator, batch_size=None, num_workers=self.num_workers,
                                           persistent_workers=self.num_workers > 0)

    def train_batch(self, batch: gd.Batch, epoch_idx: int, batch_idx: int) -> Dict[str, Any]:
        loss = None
        info = None
        try:
            loss, info = self.algo.compute_batch_losses(self.model, batch)
            if not torch.isfinite(loss):
                raise ValueError('loss is not finite')
            step_info = self.step(loss)
            if self._validate_parameters and not all([torch.isfinite(i).all() for i in self.model.parameters()]):
                raise ValueError('parameters are not finite')
        except ValueError as e:
            from ml_logger import logger
            logger.save_torch([self.model.state_dict(), batch, loss, info], 'dump.pkl')

            raise e

        if step_info is not None:
            info.update(step_info)
        if hasattr(batch, 'extra_info'):
            info.update(batch.extra_info)
        return {k: v.item() if hasattr(v, 'item') else v for k, v in info.items()}

    def evaluate_batch(self, batch: gd.Batch, epoch_idx: int = 0, batch_idx: int = 0) -> Dict[str, Any]:
        loss, info = self.algo.compute_batch_losses(self.model, batch)
        if hasattr(batch, 'extra_info'):
            info.update(batch.extra_info)
        return {k: v.item() if hasattr(v, 'item') else v for k, v in info.items()}

    def run(self):
        """Trains the GFN for `num_training_steps` minibatches, performing
        validation every `validate_every` minibatches.
        """
        from ml_logger import logger

        self.model.to(self.device)
        self.sampling_model.to(self.device)
        epoch_length = max(len(self.training_data), 1)
        train_dl = self.build_training_data_loader()
        valid_dl = self.build_validation_data_loader()
        callbacks = self.build_callbacks()
        start = getattr(self.Args, 'start_at_step', 0) + 1

        logger.print("Starting training")
        logger.start('valid_steps_timer')
        timer_steps = 0

        for it, batch in zip(range(start, 1 + self.Args.num_training_steps), cycle(train_dl)):
            epoch_idx = it // epoch_length
            batch_idx = it % epoch_length
            info = self.train_batch(batch.to(self.device), epoch_idx, batch_idx)

            logger.store_metrics(**{f'train/{k}': v for k, v in info.items()})
            timer_steps += 1

            if self.verbose:
                logger.print(f"iteration {it} : " + ' '.join(f'{k}:{v:.2f}' for k, v in info.items()))

            if it % self.Args.validate_every == 0:
                for batch in valid_dl:
                    info = self.evaluate_batch(batch.to(self.device), epoch_idx, batch_idx)
                    logger.store_metrics(**{f'valid/{k}': v for k, v in info.items()})
                    logger.print(f"validation - iteration {it} : " + ' '.join(f'{k}:{v:.2f}' for k, v in info.items()))
                end_metrics = {}
                for c in callbacks.values():
                    if hasattr(c, 'on_validation_end'):
                        c.on_validation_end(end_metrics)
                logger.store_metrics(**{f'valid_end/{k}': v for k, v in end_metrics.items()})
                logger.store_metrics(steps_per_sec=timer_steps / logger.split('valid_steps_timer'))
                timer_steps = 0

                logger.log_metrics_summary(key_values={'step': it},
                                           key_stats={
                                               'train/num_generated': 'max',
                                               'train/lifetime_hv0': 'max',
                                           })

                self._save_state(it)
        self._save_state(self.Args.num_training_steps)

    def _save_state(self, it):
        from ml_logger import logger

        logger.save_torch({
            'models_state_dict': [self.model.state_dict()],
            'Args': vars(self.Args),
            'step': it,
        }, 'checkpoints/model_state.pt')

        if it % 10_000 == 0:
            logger.save_torch({
                'models_state_dict': [self.model.state_dict()],
                'Args': vars(self.Args),
                'step': it,
            }, f'checkpoints/model_state_{it:08d}.pt')

        logger.print(f'Saved model state at step {it}. Time = {time.asctime(time.localtime())}')



def main(**deps):
    Args._update(deps)

    trainer = Trainer(Args, device=Args.device)
    trainer.run()


if __name__ == '__main__':
    from ml_logger import instr
    thunk = instr(main)
    thunk()
