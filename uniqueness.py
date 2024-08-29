"""
Calculate the uniqueness of each subject's eigenvector centrality matrix using Pearson correlation.
"""

import argparse
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from numpy.linalg import norm
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity


HDF5_FILE = "data/emodes_lh.h5" # Change to your hdf file with bids structure ./sub-x/ses-y
EIGENGROUPS = {
    "0": (0, 0),
    "1": (1, 3),
    "2": (4, 8),
    "3": (9, 15),
    "4": (16, 24),
    "5": (25, 35),
    "6": (36, 48),
    "7": (49, 63),
    "8": (64, 80),
    "9": (81, 99),
    "10": (100, 120),
    "11": (121, 143),
    "12": (144, 168),
    "13": (169, 195),
    "14": (196, 224),
}

TIMEPOINTS = [f"ses-{str(t).zfill(2)}" for t in range(1, 16)]


def align_columns(m1, m2):
    """
    Matches columns of m2 to columns of m1 based on the correlation
    """
    # Compute correlation matrix efficiently
    m1_centered = m1 - m1.mean(axis=0)
    m2_centered = m2 - m2.mean(axis=0)
    m1_std = m1_centered.std(axis=0)
    m2_std = m2_centered.std(axis=0)
    correlation_matrix = np.dot(m1_centered.T, m2_centered) / (
        m1.shape[0] * np.outer(m1_std, m2_std)
    )

    # Use the Hungarian algorithm to find the optimal column matching
    _, col_ind = linear_sum_assignment(-np.abs(correlation_matrix))

    # Reorder matrix2 columns
    m2_reordered = m2[:, col_ind]
    return m2_reordered


def calculate_similarity(matrix1, matrix2, eigen_groups=None):
    """
    Calculate Pearson correlation between two matrices,
    averaging the correlations of corresponding eigengroups.
    """

    correlations = {}
    for egroup_name, egroup in eigen_groups.items():
        # Extract the slice for current group
        cols = slice(egroup[0], egroup[1] + 1)

        submatrix1 = matrix1[:, cols]
        submatrix2 = matrix2[:, cols]

        submatrix2 = align_columns(submatrix1, submatrix2)

        # Compute the Pearson correlation matrix between corresponding columns
        correlation_matrix = np.corrcoef(submatrix1, submatrix2, rowvar=False)[
            : submatrix1.shape[1], submatrix1.shape[1] :
        ]
        # Compute mean of absolute values of diagonal elements, which are the individual column correlations
        mean_correlation = np.mean(np.abs(np.diag(correlation_matrix)))
        correlations[egroup_name] = mean_correlation

    return correlations


def process_subject(subject_id, hdf5_file, eigengroups, exclude=None):
    """
    Process the subject's eigenvector centrality matrices to calculate uniqueness
    """
    eigengroup_cols = list(eigengroups.keys())

    print(f"Processing subject {subject_id}")
    with h5py.File(hdf5_file, "r") as file:
        matrices = {
            data_set: file[data_set][:]
            for data_set in [dsname for dsname in file if subject_id in dsname]
        }
        subject_bids_datapoints = sorted(matrices.keys())
        # raise if no matrices found
        if not subject_bids_datapoints:
            raise ValueError(f"No matrices found for subject {subject_id}")

        self_similarity = pd.DataFrame(
            index=subject_bids_datapoints, columns=eigengroup_cols
        )

        self_similarity.loc[:, ["sub", "ses"]] = self_similarity.index.str.extract(
            r"sub-(?P<sub>\d+)_ses-(?P<ses>\d+)"
        ).values
        self_similarity["tp"] = self_similarity.ses.astype(int)
        self_similarity["tp_diff"] = self_similarity["tp"].diff().shift(-1)
        self_similarity = self_similarity.sort_values("tp")

        sim_to_others = {}

        for i, dp in enumerate(self_similarity.index.to_list()[:-1]):
            # print(f"Calculating self-similarity for {dp}")
            self_sim = calculate_similarity(
                matrices[dp],
                matrices[self_similarity.index[i + 1]],
                eigen_groups=eigengroups,
            )
            self_similarity.loc[dp, eigengroup_cols] = pd.Series(self_sim, name=dp)

            # Similarities to others
            next_tp = self_similarity.ses.iloc[i + 1]  # get the next timepoint

            other_ids = [
                other_id
                for other_id in file
                if other_id.endswith(next_tp) and other_id not in self_similarity.index
            ]
            sim_to_others[dp] = pd.DataFrame(index=eigengroup_cols, columns=other_ids)

            sim_to_others[dp] = sim_to_others[dp].apply(
                lambda col: pd.Series(
                    calculate_similarity(
                        matrices[dp], file[col.name][:], eigen_groups=eigengroups
                    )
                )
            )

        max_similarity_to_others = pd.concat(
            [sim_to_others[s].abs().max(axis=1) for s in sim_to_others.keys()], axis=1
        )
        max_similarity_to_others.columns = list(sim_to_others.keys())
        max_similarity_to_others = max_similarity_to_others.T
        _uniqueness = self_similarity[eigengroup_cols].div(max_similarity_to_others)

        # merge subject details to similarity to others dataframes
        max_similarity_to_others_merged = self_similarity[
            ["sub", "tp", "tp_diff"]
        ].merge(max_similarity_to_others, left_index=True, right_index=True)

        _uniqueness_merged = self_similarity[["sub", "tp", "tp_diff"]].merge(
            _uniqueness, left_index=True, right_index=True
        )

        return dict(
            uniqueness=_uniqueness_merged,
            self_similarities=self_similarity,
            max_sim_to_others=max_similarity_to_others_merged,
            similarity_to_others=sim_to_others,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate uniqueness of eigenvector matrices."
    )
    parser.add_argument(
        "hdf5_file",
        type=str,
        help="Path to the HDF5 file containing the eigenvector matrices. e.g. /data/emodes_lh.h5",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to save .csv file. e.g. /data/uniqueness/",
    )
    parser.add_argument(
        "subject_id", type=str, help="The subject ID to process. e.g. 1146"
    )

    args = parser.parse_args()

    HDF5_FILE = args.hdf5_file
    output_dir = Path(args.output_dir)
    subject_id = args.subject_id

    if not HDF5_FILE.endswith(".h5"):
        raise ValueError("The HDF5 file must have a .h5 extension.")
    if not output_dir.is_dir():
        raise ValueError("The output directory must exist.")
    if not subject_id:
        raise ValueError("The subject ID must be provided.")

    results = process_subject(subject_id, HDF5_FILE, EIGENGROUPS)
    # Save the results dict to file
    with open(output_dir / f"uniqueness_{subject_id}.pkl", "wb") as file:
        pickle.dump(results, file)
    print(f"{subject_id} uniqueness finished!")
