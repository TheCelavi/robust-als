import numpy as np
from scipy.sparse import csr_matrix

def to_sparse_matrix(df, n_users, n_items):
    df = df[df["score"] > 0]
    shape = (int(n_users), int(n_items))

    matrix = csr_matrix(
        (
            np.ones(len(df), dtype=np.float32),
            (
                df["user_idx"].to_numpy(),
                df["subject_idx"].to_numpy()
            )
        ),
        shape=shape
    )

    # Binarize
    matrix.data[:] = 1.0

    return matrix