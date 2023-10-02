import itertools
import os
import pickle

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
matplotlib.rcParams['font.serif'] = 'Computer Modern'

from sklearn.manifold import TSNE


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



def create_table(results_list, thresholds, percents=False):

    def num_combination(r_mat, t_arr, combination):
        mask = np.ones(r_mat.shape[0], dtype=bool)
        for i, t in enumerate(t_arr):
            mask &= [r_mat[:, i] < t, r_mat[:, i] >= t][combination[i]]
        return np.sum(mask)

    combinations = list(itertools.product([0, 1], repeat=len(thresholds)))
    print(thresholds)

    t_arr = np.array([t for _, t in thresholds])

    cols = ['']
    for combination in combinations:
        strs = [f'low {r_name}' if c == 0 else f'high {r_name}' for (r_name, _), c in zip(thresholds, combination)]
        cols.append('<br/>'.join(strs))

    rows = []

    for name, results in results_list:
        row = [name]

        r_ind = [results['flat_reward_names'].index(r_name) for r_name, _ in thresholds]
        r_v = results['flat_rewards'][:, r_ind]

        for i, (r_name, t_v) in enumerate(thresholds):
            print(f'{name}  R:{r_name} T:{t_v}')
            print(f'  qunatiles: {np.quantile(r_v[:, i], [0.0, 0.25, 0.5, 0.75, 1.0])}')

        row_numbers = []
        for combination in combinations:
            n = num_combination(r_v, t_arr, combination)
            row_numbers.append(n)

        if percents:
            total = np.sum(row_numbers)
            row_numbers = [int(n * 100.0 / total) for n in row_numbers]
            row_numbers[-1] = 100 - sum(row_numbers[:-1])

        row.extend([str(n) for n in row_numbers])
        rows.append(row)


    return tabulate.tabulate(rows, tablefmt='github', headers=cols)


def main():

    doc @ "# 3 distributions (independent)"

    base_thresholds = [
        ('seh', 0.5),
        ('sa', 0.6),
        ('qed', 0.25),
    ]

    palette = list(sns.color_palette("Paired"))
    sns.set_style('whitegrid')

    output_dir = os.path.basename(__file__)[:-3]
    os.makedirs(output_dir, exist_ok=True)

    doc @ "## beta = 32"
    print('beta = 32')
    results_paths = [
        ('y=SEH',
         ''),  # <seh_beta_32 eval results path>
        ('y=SA',
         ''),  # <sa_beta_32 eval results path>
        ('y=QED',
         ''),  # <qed_beta_32 eval results path>

        ('y=SEH,SA',
         ''),  # <guided(seh_beta_32, sa_beta_32, qed_beta_32, y=12) eval results path>
        ('y=SEH,QED',
         ''),  # <guided(seh_beta_32, sa_beta_32, qed_beta_32, y=13) eval results path>
        ('y=SA,QED',
         ''),  # <guided(seh_beta_32, sa_beta_32, qed_beta_32, y=23) eval results path>

        ('y=SEH,SA,QED',
         ''),  # <guided(seh_beta_32, sa_beta_32, qed_beta_32, y=123) eval results path>

        ('y=SEH,SEH',
         ''),  # <guided(seh_beta_32, sa_beta_32, qed_beta_32, y=11) eval results path>
        ('y=SA,SA',
         ''),  # <guided(seh_beta_32, sa_beta_32, qed_beta_32, y=22) eval results path>
        ('y=QED,QED',
         ''),  # <guided(seh_beta_32, sa_beta_32, qed_beta_32, y=33) eval results path>

        ('y=SEH,SEH,SEH',
         ''),  # <guided(seh_beta_32, sa_beta_32, qed_beta_32, y=111) eval results path>
        ('y=SA,SA,SA',
         ''),  # <guided(seh_beta_32, sa_beta_32, qed_beta_32, y=222) eval results path>
        ('y=QED,QED,QED',
         ''),  # <guided(seh_beta_32, sa_beta_32, qed_beta_32, y=333) eval results path>
    ]

    tex_labels = [
        '$p_\\text{SEH}$',
        '$p_\\text{SA}$',
        '$p_\\text{QED}$',

        '(a)',
        '(b)',
        '(c)',

        '(d)',

        '',
        '',
        '',

        '(e)',
        '(f)',
        '(g)',
    ]

    colors = [
        palette[0],
        palette[6],
        palette[2],

        palette[-1],
        palette[-1],
        palette[-1],
        (0.7, 0.7, 0.7),

        palette[1],
        palette[7],
        palette[3],
        palette[1],
        palette[7],
        palette[3],
    ]

    # these offsets were manually tuned for one of the TSNE plots (seed = 500)
    offsets = [
        [-25, 65],
        [-5, -65],
        [-45, 55],

        [65, 45],
        [-58, 5],
        [55, 45],
        [25, -55],

        [-20, -20],
        [20, 20],
        [-10, 15],
        [-55, -40],
        [30, -55],
        [50, 35],
    ]

    results = []
    for label, path in results_paths:
        print(label)
        load_results(path)
        results.append((label, load_results(path)))


    doc @ "### Reward stats"
    print('Reward stats')
    tbl = create_table(results, base_thresholds, percents=True)
    doc @ tbl
    doc @ ""


    doc @ "### EMD"

    if os.path.exists(f'{output_dir}/pairwise_emd.pkl'):
        with open(f'{output_dir}/pairwise_emd.pkl', 'rb') as f:
            pairwise_emd = pickle.load(f)
    else:
        pairwise_emd = get_pairwise_emd(results)
        # cache pariwise EMD in pickle file
        with open(f'{output_dir}/pairwise_emd.pkl', 'wb') as f:
            pickle.dump(pairwise_emd, f)

    headers = [''] + [label for label, _ in results]
    table = [
        [label] + [f'{x}' for x in row]
        for (label, _), row in zip(results, pairwise_emd)
    ]
    doc @ tabulate.tabulate(table, headers, tablefmt='github')

    doc @ ""

    skip_indices = [7, 8, 9]
    headers = [''] + [label for i, (label, _) in enumerate(results) if i not in skip_indices]
    table = [
        [label] + [f'{x:0.2f}' for j, x in enumerate(row) if j not in skip_indices]
        for i, ((label, _), row) in enumerate(zip(results, pairwise_emd)) if i not in skip_indices
    ]

    doc.print(tabulate.tabulate(table, headers, tablefmt='latex_booktabs'))

    doc @ ""



    print('TSNE')
    for seed in [100, 200, 300, 400, 500]:
        tsne_model = TSNE(n_components=2, perplexity=4.0,
                          early_exaggeration=100.0,
                          metric='precomputed', random_state=seed,
                          method='exact', n_iter=20000)
        embeddings = tsne_model.fit_transform(np.array(pairwise_emd))

        base_ind_1 = 0
        base_ind_2 = 1
        base_ind_3 = 2
        v_x = embeddings[base_ind_2] - embeddings[base_ind_1]
        v_x /= np.linalg.norm(v_x)
        v_y = np.array([-v_x[1], v_x[0]])
        v_y /= np.linalg.norm(v_y)
        if np.dot(v_y, embeddings[base_ind_3] - (embeddings[base_ind_1]/2.0 + embeddings[base_ind_2] / 2.0)) < 0:
            v_y *= -1
        v_mat = np.stack([v_x, v_y], axis=0)
        print(v_mat @ v_mat.T)

        embeddings = embeddings @ v_mat.T
        c = (embeddings[base_ind_1] + embeddings[base_ind_2] + embeddings[base_ind_3]) / 3.0
        embeddings -= c
        embeddings /= np.max(np.abs(embeddings))

        table = doc.table()
        row = table.row()

        fig = plt.figure(figsize=(8, 8))
        for i in range(embeddings.shape[0]):
            if i in {7, 8, 9}:
                continue
            m = 'o' if i >= 3 else 's'
            sz = 250 if i >= 3 else 300
            plt.scatter(embeddings[i, 1], embeddings[i, 0], s=sz, c=[colors[i]], marker=m,
                        edgecolors='k', linewidths=2.5, zorder=10)
        for i in range(embeddings.shape[0]):
            if i in {7, 8, 9}:
                continue
            plt.annotate(tex_labels[i], embeddings[i][::-1],
                         arrowprops=dict(arrowstyle="-|>, head_width=0.1, head_length=0.15",
                                         color='black',
                                         shrinkA=0.01, shrinkB=16.0),
                         fontsize=31, ha='center', va='center', zorder=9,
                         xytext=(offsets[i][1], offsets[i][0]), textcoords='offset points')
        plt.margins(x=0.2, y=0.3)
        plt.xticks([-0.5, 0.0, 0.5, 1.0], fontsize=18)
        plt.yticks([-0.5, 0.0, 0.5, 1.0], fontsize=18)
        plt.gca().set_aspect('equal')
        plt.grid(lw=0.5)


        plt.savefig(f'{output_dir}/embeddings_32_{seed}.pdf', bbox_inches='tight')
        row.savefig(f'{output_dir}/embeddings_32_{seed}.png',
                    bbox_inches='tight')
        plt.close()

        print(seed)

        doc @ ""

    doc.flush()


if __name__ == '__main__':
    main()
