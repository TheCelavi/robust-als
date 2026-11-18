import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f"Using device: {device}")

class GMF(nn.Module):
    def __init__(self, n_users, n_items, factors=16):
        super().__init__()
        self.user = nn.Embedding(n_users, factors)
        self.item = nn.Embedding(n_items, factors)
        self.output = nn.Linear(factors, 1)

        nn.init.normal_(self.user.weight, std=0.01)
        nn.init.normal_(self.item.weight, std=0.01)

    def forward(self, user, item):
        interaction = self.user(user) * self.item(item)
        return self.output(interaction).squeeze(-1)

    @torch.no_grad()
    def score_users(self, users):
        device = self.user.weight.device

        u = self.user(
            torch.as_tensor(users, dtype=torch.long, device=device)
        )

        weighted_items = (
                self.item.weight * self.output.weight[0]
        )

        return (
                u @ weighted_items.T + self.output.bias
        ).cpu().numpy()

def prepare_rce(X):
    X = sp.csr_matrix(X).copy()
    X.sum_duplicates()
    X.eliminate_zeros()

    n_users, n_items = X.shape
    u, i = X.nonzero()

    seen = torch.as_tensor(
        X.astype(bool).toarray(),
        dtype=torch.bool,
        device=device
    )

    return {
        "shape": X.shape,
        "pos_users": torch.as_tensor(
            u, dtype=torch.long, device=device
        ),
        "pos_items": torch.as_tensor(
            i, dtype=torch.long, device=device
        ),
        "unseen_counts": (~seen).sum(dim=1),
        "unseen_items": torch.argsort(
            seen.to(torch.uint8), dim=1
        ),
    }

def fit_rce(
    X,
    beta=0.2,
    factors=16,
    epochs=10,
    batch_size=4096,
    lr=1e-3,
    seed=42,
):

    X = sp.csr_matrix(X).copy()
    X.sum_duplicates()
    X.eliminate_zeros()

    n_users, n_items = X.shape
    u, i = X.nonzero()

    if len(u) == 0:
        raise ValueError("No positive interactions")

    torch.manual_seed(seed)
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    # Transfer training data once
    pos_users = torch.as_tensor(
        u, dtype=torch.long, device=device
    )
    pos_items = torch.as_tensor(
        i, dtype=torch.long, device=device
    )

    seen = torch.as_tensor(
        X.astype(bool).toarray(),
        dtype=torch.bool,
        device=device
    )

    # Precompute valid negative item indices.
    # Unobserved items appear first in each row.
    unseen_counts = (~seen).sum(dim=1)
    unseen_items = torch.argsort(
        seen.to(torch.uint8), dim=1
    )

    model = GMF(n_users, n_items, factors).to(device)
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr
    )

    for epoch in range(epochs):
        order = torch.randperm(
            len(pos_users),
            device=device,
            generator=rng
        )

        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            u_pos = pos_users[idx]
            i_pos = pos_items[idx]

            # Users with available negative items
            u_neg = u_pos[unseen_counts[u_pos] > 0]
            n_neg = u_neg.numel()

            if n_neg:
                ranks = (
                    torch.rand(
                        n_neg,
                        device=device,
                        generator=rng
                    ) * unseen_counts[u_neg]
                ).long()

                i_neg = unseen_items[u_neg, ranks]
            else:
                i_neg = torch.empty(
                    0, dtype=torch.long, device=device
                )

            users = torch.cat([u_pos, u_neg])
            items = torch.cat([i_pos, i_neg])

            targets = torch.cat([
                torch.ones(len(u_pos), device=device),
                torch.zeros(n_neg, device=device)
            ])

            logits = model(users, items)

            ce = F.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )

            p = logits.sigmoid().detach()

            weights = torch.where(
                targets == 1,
                p.pow(beta),
                (1 - p).pow(beta)
            )

            loss = (weights * ce).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return model


@torch.no_grad()
def recommend(model, user_id, accessible, seen_items, k=20):
    model.eval()

    device = next(model.parameters()).device

    candidates = list(set(accessible) - set(seen_items))

    if not candidates:
        return []

    items = torch.tensor(candidates, dtype=torch.long, device=device)
    users = torch.full_like(items, user_id)

    scores = model(users, items).flatten()

    indices = torch.topk(scores, k=min(k, len(candidates))).indices

    return items[indices].cpu().tolist()

