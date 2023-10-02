import os

import numpy as np
import scipy
import scipy.sparse
import tabulate

import ot

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from cmx import doc

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = (
    r'\usepackage{amsmath}'
    r'\usepackage{amssymb}'
    r'\usepackage{stix}'
    r'\newcommand{\contrast}{{\,\circlelefthalfblack\,}}'
)

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = 'Times New Roman'


def load_results(path):
    from ml_logger import ML_Logger
    loader = ML_Logger(path)

    results = loader.load_pkl('results.pkl')
    return results[0]


def num_unique_mols(mols):
    return len(set([Chem.inchi.MolToInchi(mol) for mol in mols]))


def get_pairwise_int(results):
    pairwise_int = [[0 for _ in range(len(results))] for _ in range(len(results))]
    for i, (_, results_dict1) in enumerate(results):
        pairwise_int[i][i] = num_unique_mols(results_dict1['generated_mols'])

    for i, (_, results_dict1) in enumerate(results):
        for j in range(i + 1, len(results)):
            results_dict2 = results[j][1]
            x = num_unique_mols(results_dict1['generated_mols'] + results_dict2['generated_mols'])

            pairwise_int[i][j] = pairwise_int[i][i] + pairwise_int[j][j] - x
            pairwise_int[j][i] = pairwise_int[i][j]
    return pairwise_int


def get_fingerprints(mol_list):
    dim = 2048
    row = []
    col = []
    data = []
    for i, mol in enumerate(mol_list):
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=dim)
        for j in fp.GetOnBits():
            row.append(i)
            col.append(j)
            data.append(1.0)

    sparse_fps = scipy.sparse.csr_matrix((data, (row, col)),
                                         shape=(len(mol_list), dim),
                                         dtype=np.float32)

    return sparse_fps


def get_pairwise_similarities(sparse_fps1, sparse_fps2):
    intersections = scipy.sparse.csr_matrix.dot(sparse_fps1, sparse_fps2.T)
    unions = sparse_fps1.sum(axis=1) + sparse_fps2.sum(axis=1).T - intersections
    intersections = intersections.toarray()
    unions = np.array(unions)
    unions[unions == 0] = 1
    similarities = intersections / unions
    return similarities


def get_pairwise_distances(sparse_fps1, sparse_fps2):
    eps = 1e-8
    d_max = 100.0
    similarites = get_pairwise_similarities(sparse_fps1, sparse_fps2)
    distances = 1.0 / (similarites + eps) - 1.0
    distances = np.clip(distances, a_min=0.0, a_max=d_max)

    return distances


def emd(sparse_fps1, sparse_fps2):
    distances = get_pairwise_distances(sparse_fps1, sparse_fps2)

    emd_value = ot.emd2([], [], distances)
    return emd_value


def get_pairwise_emd(results):

    sparse_fps_list = []
    for _, results_dict in results:
        sparse_fps = get_fingerprints(results_dict['generated_mols'])
        sparse_fps_list.append(sparse_fps)

    pairwise_emd = [[0 for _ in range(len(results))] for _ in range(len(results))]

    for i, sparse_fps1 in enumerate(sparse_fps_list):
        for j in range(i, len(results)):
            sparse_fps2 = sparse_fps_list[j]
            x = emd(sparse_fps1, sparse_fps2)

            pairwise_emd[i][j] = x
            if i != j:
                pairwise_emd[j][i] = x
            print(i, j)
    return pairwise_emd


def create_table(results, thresholds):
    r_ind = [results['flat_reward_names'].index(name) for name, _ in thresholds]

    r_v = results['flat_rewards'][:, r_ind]

    rows = []

    row = []
    row.append('')
    for j in [0, 1]:
        j_tag = 'low' if j == 0 else 'high'
        row.append(f'{j_tag} {thresholds[1][0]}')
    row.append('')
    row.append('sum')
    rows.append(row)

    mat = np.zeros((2, 2), dtype=np.int32)

    for i in [1, 0]:
        row = []
        i_tag = 'low' if i == 0 else 'high'
        row.append(f'{i_tag} {thresholds[0][0]}')

        low_i_mask = r_v[:, 0] < thresholds[0][1]
        i_mask = low_i_mask if i == 0 else ~low_i_mask
        for j in [0, 1]:
            low_j_mask = r_v[:, 1] < thresholds[1][1]
            j_mask = low_j_mask if j == 0 else ~low_j_mask
            num = np.sum(i_mask & j_mask)
            mat[i, j] = num

            row.append(f'{num}')
        row.append('')
        row.append(f'{np.sum(mat[i])}')
        rows.append(row)

    row = ['' for _ in range(4)]
    rows.append(row)

    row = []
    row.append('sum')
    for j in [0, 1]:
        row.append(f'{np.sum(mat[:, j])}')
    row.append('')
    row.append(f'{np.sum(mat)}')
    rows.append(row)

    return tabulate.tabulate(rows, tablefmt='github', headers='firstrow')


def make_single_result_joint_plots(label, result_dict, x_val, y_val, limits, color, title='', n_levels=4):
    flat_reward_names = result_dict['flat_reward_names']
    flat_rewards = result_dict['flat_rewards']
    joint_data = {name: flat_rewards[:, i] for i, name in enumerate(flat_reward_names)}
    g = sns.jointplot(joint_data, x=x_val, y=y_val,
                      kind='scatter', s=14, alpha=0.04,
                      xlim=limits[0], ylim=limits[1],
                      color=color,
                      marginal_ticks=True,
                      marginal_kws=dict(stat='density', alpha=0.3))
    g.plot_joint(sns.kdeplot, zorder=0,
                 color=color, n_levels=n_levels, bw_adjust=0.95,
                 alpha=0.6, linewidth=2.5)
    g.plot_marginals(sns.kdeplot, fill=True, alpha=0.3, color=color)
    plt.xlabel(x_val.upper())
    plt.ylabel(y_val.upper())
    plt.title(title, fontsize=24, y=1.23)


def make_multiple_results_joint_plots(results, x_val, y_val, limits, palette, title='', n_levels=4,
                                      scatter_alpha=0.04, extra_kde_results=(), extra_kde_colors=()):
    joint_data = dict()
    for label, result in results:
        label_array = np.zeros((0,), dtype=str)
        if 'label' in joint_data:
            label_array = joint_data['label']

        for i, name in enumerate(result['flat_reward_names']):
            val_array = np.zeros((0,))
            if name in joint_data:
                val_array = joint_data[name]
            val_array = np.concatenate((val_array, result['flat_rewards'][:, i]))

            joint_data[name] = val_array

        label_array = np.concatenate((label_array, np.full((result['flat_rewards'].shape[0],), label)))
        joint_data['label'] = label_array

    g = sns.jointplot(joint_data, x=x_val, y=y_val,
                      hue='label', palette=palette[:len(results)],
                      kind='scatter', s=14, alpha=scatter_alpha,
                      xlim=limits[0], ylim=limits[1],
                      )
    g.plot_joint(sns.kdeplot, zorder=2,
                 n_levels=n_levels, bw_adjust=0.95,
                 fill=False, alpha=0.6, linewidths=2.5)

    g.ax_marg_x.remove()
    g.ax_marg_y.remove()

    for i, (extra_kde_result, extra_kde_color) in enumerate(zip(extra_kde_results, extra_kde_colors)):

        flat_reward_names = extra_kde_result[1]['flat_reward_names']
        extra_joint_data = {name: extra_kde_result[1]['flat_rewards'][:, i] for i, name in enumerate(flat_reward_names)}
        sns.kdeplot(extra_joint_data, x=x_val, y=y_val, ax=g.ax_joint,
                    zorder=0, color=extra_kde_color, n_levels=n_levels, bw_adjust=0.95,
                    alpha=0.8, linewidths=1)

    plt.xlabel(x_val.upper(), fontsize=32)
    plt.ylabel(y_val.upper(), fontsize=32)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.grid(lw=0.5)

    lgnd = plt.legend(loc='upper center', bbox_to_anchor=(0.4, 1.2),
                      ncol=len(results), frameon=False,
                      columnspacing=-0.1, handletextpad=-0.55, labelspacing=-2.0,
                      fontsize=28)
    for i in range(len(lgnd.legend_handles)):
        lgnd.legend_handles[i]._sizes = [180]


def main():

    doc @ "# Molecules 2 distributions"

    x_val = 'seh'
    y_val = 'sa'
    limits = ((-0.02, 1.1), (0.4, 0.95))
    palette = list(sns.color_palette("Paired")) + ['#888888']
    sns.set_style('whitegrid')

    output_dir = os.path.basename(__file__)[:-3]
    os.makedirs(output_dir, exist_ok=True)

    doc @ "## beta = 32"
    print('beta = 32')
    results_paths = [
        ('SEH',
         ''),  # <seh_beta_32 eval results path>
        ('SA',
         ''),  # <sa_beta_32 eval results path>
        ('C(SEH, SA, 0.05)',
         ''),  # <guided(seh_beta_32, sa_beta_32, y=11, alpha=0.05) eval results path>
        ('C(SA, SEH, 0.05)',
         ''),  # <guided(seh_beta_32, sa_beta_32, y=22, alpha=0.95) eval results path>
        ('HM(SA, SEH, 0.50)',
         ''),  # <guided(seh_beta_32, sa_beta_32, y=12, alpha=0.50) eval results path>
    ]

    tex_labels = [
        '$p_\\text{SEH}$',
        '$p_\\text{SA}$',
        '$p_\\text{SEH} \\contrast_{\\! \\scriptscriptstyle 0.95}\\;p_\\text{SA}$',
        '$p_\\text{SA} \\contrast_{\\! \\scriptscriptstyle 0.95}\\;p_\\text{SEH}$',
        '$p_\\text{SEH} \\otimes p_\\text{SA}$',
    ]

    results = []
    for label, path in results_paths:
        load_results(path)
        results.append((label, load_results(path)))

    results_with_tex = [
        (tex_label, result) for tex_label, (_, result) in zip(tex_labels, results)
    ]

    doc @ "### Figures"

    table = doc.table()
    row = table.row()

    fig = plt.figure(figsize=(10, 8))
    make_multiple_results_joint_plots(results_with_tex[:2][::-1],
                                      x_val, y_val,
                                      limits, [palette[1], palette[7]][::-1],
                                      title='Base GFlowNets')
    plt.savefig(f'{output_dir}/base_gflownets_32.pdf', bbox_inches='tight')
    row.savefig(f'{output_dir}/base_gflownets_32.png', dpi=300,
                bbox_inches='tight')
    plt.close()

    row = table.row()
    fig = plt.figure(figsize=(10, 8))
    make_multiple_results_joint_plots((results_with_tex[2:3] + results_with_tex[3:4])[::-1],
                                      x_val, y_val,
                                      limits, [palette[9], palette[5]][::-1],
                                      title='Contrasts')
    plt.savefig(f'{output_dir}/contrasts_005_32.pdf', bbox_inches='tight')
    row.savefig(f'{output_dir}/contrasts_005_32.png', dpi=300,
                bbox_inches='tight')
    plt.close()


    row = table.row()
    fig = plt.figure(figsize=(10, 8))
    make_multiple_results_joint_plots(results_with_tex[4:5],
                                      x_val, y_val,
                                      limits, palette[3:4],
                                      title='HM',
                                      extra_kde_results=results_with_tex[:2],
                                      extra_kde_colors=[palette[1], palette[7]])
    plt.savefig(f'{output_dir}/hm_050_32.pdf', bbox_inches='tight')
    row.savefig(f'{output_dir}/hm_050_32.png', dpi=300,
                bbox_inches='tight')
    plt.close()

    doc @ ""

    doc.flush()


if __name__ == '__main__':
    main()
