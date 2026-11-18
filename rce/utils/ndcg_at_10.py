import numpy as np


def ndcg_at_10(model, X_train, X_valid, k=10, min_train=10):
    X_train = X_train.tocsr()
    X_valid = X_valid.tocsr()

    if X_train.shape != X_valid.shape:
        raise ValueError(
            "Train and validation must share user/item indices."
        )

    users = np.flatnonzero(
        (X_train.getnnz(axis=1) >= min_train) &
        (X_valid.getnnz(axis=1) > 0)
    )

    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    ndcgs = []

    for offset in range(0, len(users), 128):
        batch_users = users[offset:offset + 128]
        scores = model.score_users(batch_users)

        # Exclude training interactions
        seen = X_train[batch_users].tocoo()
        scores[seen.row, seen.col] = -np.inf

        k_eff = min(k, scores.shape[1])

        candidate_idx = np.argpartition(
            -scores,
            kth=k_eff - 1,
            axis=1
        )[:, :k_eff]

        candidate_scores = np.take_along_axis(
            scores, candidate_idx, axis=1
        )

        order = np.argsort(-candidate_scores, axis=1)

        topk = np.take_along_axis(
            candidate_idx, order, axis=1
        )

        for j, user in enumerate(batch_users):
            relevant = X_valid[user].indices

            # Remove overlaps with training
            relevant = relevant[
                np.isfinite(scores[j, relevant])
            ]

            if len(relevant) == 0:
                continue

            hits = np.isin(topk[j], relevant)

            dcg = (
                    hits * discounts[:len(hits)]
            ).sum()

            idcg = discounts[
                :min(len(relevant), k)
            ].sum()

            ndcgs.append(dcg / idcg)

    if not ndcgs:
        raise ValueError("No eligible validation users.")

    return float(np.mean(ndcgs))
