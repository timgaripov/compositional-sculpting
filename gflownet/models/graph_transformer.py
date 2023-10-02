import math
import torch
import torch.nn as nn
import torch_geometric.data as gd
import torch_geometric.nn as gnn
from torch_geometric.utils import add_self_loops

from gflownet.envs.graph_building_env import GraphActionCategorical
from gflownet.envs.graph_building_env import GraphActionType


def mlp(n_in, n_hid, n_out, n_layer, act=nn.LeakyReLU):
    """Creates a fully-connected network with no activation after the last layer.
    If `n_layer` is 0 then this corresponds to `nn.Linear(n_in, n_out)`.
    """
    n = [n_in] + [n_hid] * n_layer + [n_out]
    return nn.Sequential(*sum([[nn.Linear(n[i], n[i + 1]), act()] for i in range(n_layer + 1)], [])[:-1])


class GraphTransformer(nn.Module):
    """An agnostic GraphTransformer class, and the main model used by other model classes

    This graph model takes in node features, edge features, and graph features (referred to as
    conditional information, since they condition the output). The graph features are projected to
    virtual nodes (one per graph), which are fully connected.

    The per node outputs are the concatenation of the final (post graph-convolution) node embeddings
    and of the final virtual node embedding of the graph each node corresponds to.

    The per graph outputs are the concatenation of a global mean pooling operation, of the final
    virtual node embeddings, and of the conditional information embedding.
    """
    def __init__(self, x_dim, e_dim, g_dim, num_emb=64, num_layers=3, num_heads=2, num_noise=0, ln_type='pre'):
        """
        Parameters
        ----------
        x_dim: int
            The number of node features
        e_dim: int
            The number of edge features
        g_dim: int
            The number of graph-level features
        num_emb: int
            The number of hidden dimensions, i.e. embedding size. Default 64.
        num_layers: int
            The number of Transformer layers.
        num_heads: int
            The number of Transformer heads per layer.
        ln_type: str
            The location of Layer Norm in the transformer, either 'pre' or 'post', default 'pre'.
            (apparently, before is better than after, see https://arxiv.org/pdf/2002.04745.pdf)
        """
        super().__init__()
        self.num_layers = num_layers
        self.num_noise = num_noise
        assert ln_type in ['pre', 'post']
        self.ln_type = ln_type

        self.x2h = mlp(x_dim + num_noise, num_emb, num_emb, 2)
        self.e2h = mlp(e_dim, num_emb, num_emb, 2)
        self.c2h = mlp(g_dim, num_emb, num_emb, 2)
        self.graph2emb = nn.ModuleList(
            sum([[
                gnn.GENConv(num_emb, num_emb, num_layers=1, aggr='add', norm=None),
                gnn.TransformerConv(num_emb * 2, num_emb, edge_dim=num_emb, heads=num_heads),
                nn.Linear(num_heads * num_emb, num_emb),
                gnn.LayerNorm(num_emb, affine=False),
                mlp(num_emb, num_emb * 4, num_emb, 1),
                gnn.LayerNorm(num_emb, affine=False),
                nn.Linear(num_emb, num_emb * 2),
            ] for i in range(self.num_layers)], []))

    def forward(self, g: gd.Batch, cond: torch.Tensor):
        """Forward pass

        Parameters
        ----------
        g: gd.Batch
            A standard torch_geometric Batch object. Expects `edge_attr` to be set.
        cond: torch.Tensor
            The per-graph conditioning information. Shape: (g.num_graphs, self.g_dim).

        Returns
        node_embeddings: torch.Tensor
            Per node embeddings. Shape: (g.num_nodes, self.num_emb).
        graph_embeddings: torch.Tensor
            Per graph embeddings. Shape: (g.num_graphs, self.num_emb * 2).
        """
        if self.num_noise > 0:
            x = torch.cat([g.x, torch.rand(g.x.shape[0], self.num_noise, device=g.x.device)], 1)
        else:
            x = g.x
        o = self.x2h(x)
        e = self.e2h(g.edge_attr)
        c = self.c2h(cond)
        num_total_nodes = g.x.shape[0]
        # Augment the edges with a new edge to the conditioning
        # information node. This new node is connected to every node
        # within its graph.
        u, v = torch.arange(num_total_nodes, device=o.device), g.batch + num_total_nodes
        aug_edge_index = torch.cat([g.edge_index, torch.stack([u, v]), torch.stack([v, u])], 1)
        e_p = torch.zeros((num_total_nodes * 2, e.shape[1]), device=g.x.device)
        e_p[:, 0] = 1  # Manually create a bias term
        aug_e = torch.cat([e, e_p], 0)
        aug_edge_index, aug_e = add_self_loops(aug_edge_index, aug_e, 'mean')
        aug_batch = torch.cat([g.batch, torch.arange(c.shape[0], device=o.device)], 0)

        # Append the conditioning information node embedding to o
        o = torch.cat([o, c], 0)
        for i in range(self.num_layers):
            # Run the graph transformer forward
            gen, trans, linear, norm1, ff, norm2, cscale = self.graph2emb[i * 7:(i + 1) * 7]
            cs = cscale(c[aug_batch])
            if self.ln_type == 'post':
                agg = gen(o, aug_edge_index, aug_e)
                l_h = linear(trans(torch.cat([o, agg], 1), aug_edge_index, aug_e))
                scale, shift = cs[:, :l_h.shape[1]], cs[:, l_h.shape[1]:]
                o = norm1(o + l_h * scale + shift, aug_batch)
                o = norm2(o + ff(o), aug_batch)
            else:
                o_norm = norm1(o, aug_batch)
                agg = gen(o_norm, aug_edge_index, aug_e)
                l_h = linear(trans(torch.cat([o_norm, agg], 1), aug_edge_index, aug_e))
                scale, shift = cs[:, :l_h.shape[1]], cs[:, l_h.shape[1]:]
                o = o + l_h * scale + shift
                o = o + ff(norm2(o, aug_batch))

        glob = torch.cat([gnn.global_mean_pool(o[:-c.shape[0]], g.batch), o[-c.shape[0]:]], 1)
        o_final = torch.cat([o[:-c.shape[0]]], 1)
        return o_final, glob


class GraphTransformerGFN(nn.Module):
    """GraphTransformer class for a GFlowNet which outputs a GraphActionCategorical. Meant for atom-wise
    generation.

    Outputs logits for the following actions
    - Stop
    - AddNode
    - SetNodeAttr
    - AddEdge
    - SetEdgeAttr

    """
    def __init__(self, env_ctx, num_emb=64, num_layers=3, num_heads=2, num_mlp_layers=0):
        """See `GraphTransformer` for argument values"""
        super().__init__()
        self.transf = GraphTransformer(x_dim=env_ctx.num_node_dim, e_dim=env_ctx.num_edge_dim,
                                       g_dim=env_ctx.num_cond_dim, num_emb=num_emb, num_layers=num_layers,
                                       num_heads=num_heads)
        num_final = num_emb
        num_glob_final = num_emb * 2
        num_edge_feat = num_emb if env_ctx.edges_are_unordered else num_emb * 2
        self.edges_are_duplicated = env_ctx.edges_are_duplicated
        self.edges_are_unordered = env_ctx.edges_are_unordered

        self.emb2add_edge = mlp(num_edge_feat, num_emb, 1, num_mlp_layers)
        self.emb2add_node = mlp(num_final, num_emb, env_ctx.num_new_node_values, num_mlp_layers)
        if env_ctx.num_node_attr_logits is not None:
            self.emb2set_node_attr = mlp(num_final, num_emb, env_ctx.num_node_attr_logits, num_mlp_layers)
        if env_ctx.num_edge_attr_logits is not None:
            self.emb2set_edge_attr = mlp(num_edge_feat, num_emb, env_ctx.num_edge_attr_logits, num_mlp_layers)
        self.emb2stop = mlp(num_glob_final, num_emb, 1, num_mlp_layers)
        self.emb2reward = mlp(num_glob_final, num_emb, 1, num_mlp_layers)
        self.logZ = mlp(env_ctx.num_cond_dim, num_emb * 2, 1, 2)
        self.action_type_order = env_ctx.action_type_order

        self._emb_to_logits = {
            GraphActionType.Stop: lambda emb: self.emb2stop(emb['graph']),
            GraphActionType.AddNode: lambda emb: self.emb2add_node(emb['node']),
            GraphActionType.SetNodeAttr: lambda emb: self.emb2set_node_attr(emb['node']),
            GraphActionType.AddEdge: lambda emb: self.emb2add_edge(emb['non_edge']),
            GraphActionType.SetEdgeAttr: lambda emb: self.emb2set_edge_attr(emb['edge']),
        }
        self._action_type_to_key = {
            GraphActionType.Stop: None,
            GraphActionType.AddNode: 'x',
            GraphActionType.SetNodeAttr: 'x',
            GraphActionType.AddEdge: 'non_edge_index',
            GraphActionType.SetEdgeAttr: 'edge_index'
        }
        self._action_type_to_mask_name = {
            GraphActionType.Stop: 'stop',
            GraphActionType.AddNode: 'add_node',
            GraphActionType.SetNodeAttr: 'set_node_attr',
            GraphActionType.AddEdge: 'add_edge',
            GraphActionType.SetEdgeAttr: 'set_edge_attr'
        }

    def _action_type_to_mask(self, t, g):
        mask_name = self._action_type_to_mask_name[t] + '_mask'
        return getattr(g, mask_name) if hasattr(g, mask_name) else 1

    def _action_type_to_logit(self, t, emb, g):
        return self._mask(self._emb_to_logits[t](emb), self._action_type_to_mask(t, g))

    def _mask(self, x, m):
        # mask logit vector x with binary mask m, -1000 is a tiny log-value
        return x * m + -1000 * (1 - m)

    def forward(self, g: gd.Batch, cond: torch.Tensor):
        node_embeddings, graph_embeddings = self.transf(g, cond)
        # "Non-edges" are edges not currently in the graph that we could add
        if hasattr(g, 'non_edge_index'):
            ne_row, ne_col = g.non_edge_index
            if self.edges_are_unordered:
                non_edge_embeddings = node_embeddings[ne_row] + node_embeddings[ne_col]
            else:
                non_edge_embeddings = torch.cat([node_embeddings[ne_row], node_embeddings[ne_col]], 1)
        else:
            # If the environment context isn't setting non_edge_index, we can safely assume that
            # action is not in ctx.action_type_order.
            non_edge_embeddings = None
        if self.edges_are_duplicated:
            # On `::2`, edges are typically duplicated to make graphs undirected, only take the even ones
            e_row, e_col = g.edge_index[:, ::2]
        else:
            e_row, e_col = g.edge_index
        if self.edges_are_unordered:
            edge_embeddings = node_embeddings[e_row] + node_embeddings[e_col]
        else:
            edge_embeddings = torch.cat([node_embeddings[e_row], node_embeddings[e_col]], 1)

        emb = {
            'graph': graph_embeddings,
            'node': node_embeddings,
            'edge': edge_embeddings,
            'non_edge': non_edge_embeddings,
        }

        cat = GraphActionCategorical(
            g,
            logits=[self._action_type_to_logit(t, emb, g) for t in self.action_type_order],
            keys=[self._action_type_to_key[t] for t in self.action_type_order],
            masks=[self._action_type_to_mask(t, g) for t in self.action_type_order],
            types=self.action_type_order,
        )
        return cat, self.emb2reward(graph_embeddings)


class GraphTransformerClassifier(nn.Module):
    """GraphTransformer class for a graph classifier
    """
    def __init__(self, env_ctx, num_cond=1, num_emb=64, num_layers=3, num_heads=2, num_mlp_layers=0):
        """See `GraphTransformer` for argument values"""
        super().__init__()
        self.transf = GraphTransformer(x_dim=env_ctx.num_node_dim, e_dim=env_ctx.num_edge_dim,
                                       g_dim=num_cond, num_emb=num_emb, num_layers=num_layers,
                                       num_heads=num_heads)
        num_glob_final = num_emb * 2
        self.emb2logits = mlp(num_glob_final, num_emb, 1, num_mlp_layers)

    def forward(self, g: gd.Batch, terminal: torch.Tensor):
        _, graph_embeddings = self.transf(g, terminal)
        logits = self.emb2logits(graph_embeddings).squeeze(dim=-1)
        return logits


class GraphTransformerJointClassifier(nn.Module):
    """GraphTransformer class for a graph classifier
    """
    def __init__(self, env_ctx, num_cond=1, num_emb=64, num_layers=3, num_heads=2, num_mlp_layers=0):
        """See `GraphTransformer` for argument values"""
        super().__init__()
        self.transf = GraphTransformer(x_dim=env_ctx.num_node_dim, e_dim=env_ctx.num_edge_dim,
                                       g_dim=num_cond, num_emb=num_emb, num_layers=num_layers,
                                       num_heads=num_heads)
        num_glob_final = num_emb * 2
        self.non_term_head = mlp(num_glob_final, num_emb, 2, num_mlp_layers)
        self.term_head = mlp(num_glob_final, num_emb, 1, num_mlp_layers)


    def forward(self, g: gd.Batch, terminal: torch.Tensor):
        _, graph_embeddings = self.transf(g, terminal)

        non_term_outputs = self.non_term_head(graph_embeddings)
        term_outputs = self.term_head(graph_embeddings)

        # log_probs shape [batch_size, 2x2]
        # non-term probs:
        # p(y_1=1, y_2=1) = a
        # p(y_1=2, y_2=2) = b
        # p(y_1=1, y_2=2) = p(y_1=2, y_2=1) = c
        # a + b + 2c = 1
        # log(a + b + 2c) = 0
        # a = exp(o_0) / (exp(o_0) + exp(o_1) + 2 * 1)
        # b = exp(o_1) / (exp(o_0) + exp(o_1) + 2 * 1)
        # c = 1 / (exp(o_0) + exp(o_1) + 2 * 1)
        non_term_tmp = torch.cat([non_term_outputs, torch.full_like(non_term_outputs[:, :1], math.log(2.0))], dim=1)
        non_term_tmp = torch.log_softmax(non_term_tmp, dim=1)
        non_term_log_probs = torch.cat([non_term_tmp[:, :1], non_term_tmp[:, 2:] - math.log(2.0),
                                        non_term_tmp[:, 2:] - math.log(2.0), non_term_tmp[:, 1:2]], dim=1)

        # term probs:
        # p(y_1 = 1) = p(y_2 = 1) = a
        # p(y_1 = 2) = p(y_2 = 2) = b

        # p(y_1 = 1, y_2 = 2) = a^2
        # p(y_1 = 2, y_2 = 2) = b^2
        # p(y_1 = 1, y_2 = 1) = ab
        # p(y_1 = 2, y_2 = 1) = ab

        # log p(y_1 = 1, y_2 = 1) = 2 * log a
        # log p(y_1 = 2, y_2 = 2) = 2 * log b
        # log p(y_1 = 1, y_2 = 2) = log a + log b
        # log p(y_1 = 2, y_2 = 1) = log a + log b

        term_log_a = torch.nn.functional.logsigmoid(-term_outputs)
        term_log_b = torch.nn.functional.logsigmoid(term_outputs)

        term_log_ab = torch.cat([term_log_a, term_log_b], dim=1)

        term_log_probs = (term_log_ab[:, :, None] + term_log_ab[:, None, :]).view(-1, 4)

        log_probs = non_term_log_probs * (1.0 - terminal.view(-1, 1)) + term_log_probs * terminal.view(-1, 1)
        log_probs = log_probs.view(-1, 2, 2)

        return log_probs


class GraphTransformerJointClassifierParam(nn.Module):
    """GraphTransformer class for a graph classifier
    """
    def __init__(self, env_ctx, num_cond=1, num_emb=64, num_layers=3, num_heads=2, num_mlp_layers=0):
        """See `GraphTransformer` for argument values"""
        super().__init__()
        self.transf = GraphTransformer(x_dim=env_ctx.num_node_dim, e_dim=env_ctx.num_edge_dim,
                                       g_dim=num_cond, num_emb=num_emb, num_layers=num_layers,
                                       num_heads=num_heads)
        num_glob_final = num_emb * 2
        self.non_term_head = mlp(num_glob_final + 1, num_emb, 3, num_mlp_layers)
        self.term_head = mlp(num_glob_final, num_emb, 1, num_mlp_layers)

    def get_outputs(self, g, logit_alpha, terminal):
        # g: batch_size gd.Batch
        # logit_alpha [batch_size, 1]
        # terminal: [batch_size, 1] 0.0 or 1.0
        cond = logit_alpha * (1.0 - terminal)

        _, graph_embeddings = self.transf(g, cond)

        non_term_outputs = self.non_term_head(torch.cat((graph_embeddings, cond), dim=1))
        term_outputs = self.term_head(graph_embeddings)

        return non_term_outputs, term_outputs

    def forward(self, g: gd.Batch, logit_alpha: torch.Tensor, terminal: torch.Tensor):
        # logit_alpha [batch_size, 1]
        # terminal: [batch_size, 1] 0.0 or 1.0

        non_term_outputs, term_outputs = self.get_outputs(g, logit_alpha, terminal)

        # log_probs shape [batch_size, 2x2]

        non_term_tmp = torch.cat([non_term_outputs, torch.zeros_like(non_term_outputs[:, :1])], dim=1)
        # [batch_size, 4]
        non_term_log_probs = torch.log_softmax(non_term_tmp, dim=1)

        # term probs:
        # p(y_1 = 1) = a
        # p(y_1 = 2) = b

        # p(y_2 = 1) = c
        # p(y_2 = 2) = d

        # p(y_1 = 1, y_2 = 1) = ac
        # p(y_1 = 2, y_2 = 2) = bd
        # p(y_1 = 1, y_2 = 2) = ad
        # p(y_1 = 2, y_2 = 1) = bc

        # log p(y_1 = 1, y_2 = 1) = log a + log c
        # log p(y_1 = 2, y_2 = 2) = log b + log d
        # log p(y_1 = 1, y_2 = 2) = log a + log d
        # log p(y_1 = 2, y_2 = 1) = log b + log c

        term_log_a = torch.nn.functional.logsigmoid(-term_outputs)
        term_log_b = torch.nn.functional.logsigmoid(term_outputs)
        term_log_c = torch.nn.functional.logsigmoid(-(term_outputs - logit_alpha))
        term_log_d = torch.nn.functional.logsigmoid(term_outputs - logit_alpha)

        term_log_ab = torch.cat([term_log_a, term_log_b], dim=1)
        term_log_cd = torch.cat([term_log_c, term_log_d], dim=1)

        term_log_probs = (term_log_ab[:, :, None] + term_log_cd[:, None, :]).view(-1, 4)

        log_probs = non_term_log_probs * (1.0 - terminal.view(-1, 1)) + term_log_probs * terminal.view(-1, 1)
        log_probs = log_probs.view(-1, 2, 2)

        return log_probs


class GraphTransformerJointClassifier3D3Y(nn.Module):
    """GraphTransformer class for a graph classifier
    """
    def __init__(self, env_ctx, num_cond=1, num_emb=64, num_layers=3, num_heads=2, num_mlp_layers=0):
        """See `GraphTransformer` for argument values"""
        super().__init__()
        self.transf = GraphTransformer(x_dim=env_ctx.num_node_dim, e_dim=env_ctx.num_edge_dim,
                                       g_dim=num_cond, num_emb=num_emb, num_layers=num_layers,
                                       num_heads=num_heads)
        num_glob_final = num_emb * 2
        self.non_term_head = mlp(num_glob_final, num_emb, 9, num_mlp_layers)
        self.term_head = mlp(num_glob_final, num_emb, 2, num_mlp_layers)


    def forward(self, g: gd.Batch, terminal: torch.Tensor):
        _, graph_embeddings = self.transf(g, terminal)

        non_term_outputs = self.non_term_head(graph_embeddings)
        term_outputs = self.term_head(graph_embeddings)

        # log_probs shape [batch_size, 3x3x3]
        # non-term probs:
        # a + b + c + 3d + 3e + 3f + 3g + 3h + 3i + 6k = 1

        non_term_tmp = torch.cat([non_term_outputs, torch.zeros_like(non_term_outputs[:, :1])], dim=1)
        non_term_tmp = torch.log_softmax(non_term_tmp, dim=1)
        # [batch_size, 10]

        aslice = non_term_tmp[:, :1]
        bslice = non_term_tmp[:, 1:2]
        cslice = non_term_tmp[:, 2:3]
        dslice = non_term_tmp[:, 3:4] - math.log(3.0)
        eslice = non_term_tmp[:, 4:5] - math.log(3.0)
        fslice = non_term_tmp[:, 5:6] - math.log(3.0)
        gslice = non_term_tmp[:, 6:7] - math.log(3.0)
        hslice = non_term_tmp[:, 7:8] - math.log(3.0)
        islice = non_term_tmp[:, 8:9] - math.log(3.0)
        kslice = non_term_tmp[:, 9:10] - math.log(6.0)

        non_term_log_probs = torch.cat([
            aslice,  # 111
            dslice,  # 112
            eslice,  # 113
            dslice,  # 121
            fslice,  # 122,
            kslice,  # 123
            eslice,  # 131
            kslice,  # 132
            hslice,  # 133

            dslice,  # 211
            fslice,  # 212
            kslice,  # 213
            fslice,  # 221
            bslice,  # 222
            gslice,  # 223
            kslice,  # 231
            gslice,  # 232
            islice,  # 233

            eslice,  # 311
            kslice,  # 312
            hslice,  # 313
            kslice,  # 321
            gslice,  # 322
            islice,  # 323
            hslice,  # 331
            islice,  # 332
            cslice,  # 333
        ], dim=1)

        # term probs:
        # p(y_1 = 1) = p(y_2 = 1) = a
        # p(y_1 = 2) = p(y_2 = 2) = b
        # p(y_1 = 3) = p(y_2 = 3) = c

        term_logp_single = torch.log_softmax(
            torch.cat([term_outputs, torch.zeros_like(term_outputs[:, :1])], dim=1), dim=1)

        term_log_probs = (
            term_logp_single[:, :, None, None] +
            term_logp_single[:, None, :, None] +
            term_logp_single[:, None, None, :]
        ).view(-1, 27)


        log_probs = non_term_log_probs * (1.0 - terminal.view(-1, 1)) + term_log_probs * terminal.view(-1, 1)
        log_probs = log_probs.view(-1, 3, 3, 3)

        return log_probs
