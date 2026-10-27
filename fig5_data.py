"""
Figure 5 data: everything CALCULATED to illustrate how the communication-subspace
null is built (the logic of CCA_null_dist_test.py) for one example condition.

Example condition: Primary -> Dorsal vs Primary -> Ventral, natural sounds,
sustained response window (set in CONDITION below). For one example session this
re-fits the two CCAs and saves what the real-data panels need; the pooled
z-score panel uses every session of the condition. Writes to
figure_params.figure_data_dir("figure5"); fig5_figures.py only reads it.

The method being illustrated
  U, V   = Primary-side CCA weights for Primary->target1 and Primary->target2
           (n_neurons x d each): the two "communication subspaces" in Primary's
           own neuron space.
  R      = ||U^T V||_F^2 / min(||U||_F^2, ||V||_F^2), the overlap of the subspaces.
  Null   = draw d vectors from N(0, Sigma) (Sigma = covariance of the Primary
           data), orthonormalize with QR, do it twice, compute R. Repeat.
  z      = (R_actual - mean(null)) / std(null).
"""
import os
import pickle

import numpy as np
from sklearn.cross_decomposition import CCA

import figure_params as fp

FIG = "figure5"

CONDITION = dict(source="Primary auditory area",
                 target1="Dorsal auditory area",
                 target2="Ventral auditory area",
                 stimulus="naturalSound",
                 response_range="sustained")
EXAMPLE_SESSION = None       # date string; None = first session of the condition

NEURON_THRESHOLD = 40        # as in CCA_null_dist_test.__main__
CCA_RANDOM_STATE = 42
SEED = 0
N_NULL_ILLUSTRATED = 8       # null pairs kept for the "null draw" panel
N_NULL_ANGLES = 2000         # draws used for the principal-angle band
N_NULL_ISOTROPIC = 10000     # isotropic (cov = I) null, for comparison


# ----------------------------------------------------------------------------
# Subspace geometry (pure numpy)
# ----------------------------------------------------------------------------
def subspace_overlap(U, V):
    """R exactly as in CCA_null_dist_test.py."""
    return np.linalg.norm(U.T @ V) ** 2 / min(np.linalg.norm(U) ** 2, np.linalg.norm(V) ** 2)


def orthonormalize(A):
    return np.linalg.qr(A)[0]


def principal_cosines(Qu, Qv):
    """Cosines of the principal angles between two orthonormal bases, descending."""
    return np.linalg.svd(Qu.T @ Qv, compute_uv=False)


def _cov_sqrt(cov):
    evals, evecs = np.linalg.eigh(cov)
    return evecs * np.sqrt(np.clip(evals, 0, None))


def sample_null_bases(root, d_u, d_v, rng):
    """d_u and d_v independent draws from N(0, Sigma) (root = _cov_sqrt(Sigma), or
    the identity for the isotropic null); returns the raw samples and their QR bases."""
    u_hat = (rng.standard_normal((d_u, root.shape[0])) @ root.T).T   # (n, d_u)
    v_hat = (rng.standard_normal((d_v, root.shape[0])) @ root.T).T
    return u_hat, v_hat, orthonormalize(u_hat), orthonormalize(v_hat)


def compute_example(X, Y1, Y2, d_u, d_v, seed=SEED):
    """
    X: (nTrials, nSourceNeurons); Y1, Y2: target regions (nTrials, n). Re-fits the
    two CCAs and returns everything the real-data panels need.
    """
    rng = np.random.default_rng(seed)
    U = CCA(n_components=d_u, max_iter=1000).fit(X, Y1).x_weights_   # (n, d_u)
    V = CCA(n_components=d_v, max_iter=1000).fit(X, Y2).x_weights_
    Qu, Qv = orthonormalize(U), orthonormalize(V)

    cov = np.cov(X.T)
    evals, evecs = np.linalg.eigh(cov)
    evals, evecs = evals[::-1], evecs[:, ::-1]
    pcs = evecs[:, :2]                                   # PC1, PC2 of Primary
    root = _cov_sqrt(cov)
    n = cov.shape[0]

    def shadow(Q):                                       # 2 x d: Q's footprint in the PC1-PC2 plane
        return pcs.T @ Q

    out = dict(
        d_u=d_u, d_v=d_v, n_neurons=n,
        R_actual=subspace_overlap(U, V),
        R_actual_orthonormalized=subspace_overlap(Qu, Qv),
        cos_actual=principal_cosines(Qu, Qv),
        pc_evals=evals[:2], pc_var_frac=evals[:2] / evals.sum(),
        scores=(X - X.mean(0)) @ pcs,
        shadow_u=shadow(Qu), shadow_v=shadow(Qv),
    )

    illustrated = []
    for k in range(N_NULL_ILLUSTRATED):
        u_hat, v_hat, Qun, Qvn = sample_null_bases(root, d_u, d_v, rng)
        item = dict(shadow_u=shadow(Qun), shadow_v=shadow(Qvn),
                    R=subspace_overlap(Qun, Qvn))
        if k == 0:  # the raw samples, before QR, as points in the PC plane
            item.update(samples_u=u_hat.T @ pcs, samples_v=v_hat.T @ pcs)
        illustrated.append(item)
    out["null_illustrated"] = illustrated

    d_min = min(d_u, d_v)
    cos_null = np.empty((N_NULL_ANGLES, d_min))
    for k in range(N_NULL_ANGLES):
        _, _, Qun, Qvn = sample_null_bases(root, d_u, d_v, rng)
        cos_null[k] = principal_cosines(Qun, Qvn)
    out["cos_null_mean"] = cos_null.mean(0)
    out["cos_null_lo"], out["cos_null_hi"] = np.percentile(cos_null, [5, 95], axis=0)

    eye = np.eye(n)
    out["R_null_isotropic"] = np.array([
        subspace_overlap(*sample_null_bases(eye, d_u, d_v, rng)[2:]) for _ in range(N_NULL_ISOTROPIC)])
    return out


def pooled_from_rows(rows):
    """Per-row z-scored nulls and actual z for the pooled histogram (rows = feather rows)."""
    pooled = []
    for r in rows:
        null = np.asarray(r["null_similarities"], dtype=float)
        sd = null.std(ddof=1)
        pooled.append(dict(session=str(r["session"]), R_actual=float(r["R_actual"]),
                           null_R=null.astype(np.float32),
                           z_null=((null - null.mean()) / sd if sd > 0 else np.zeros_like(null)).astype(np.float32),
                           z_actual=float(r["z_score_diff_from_null"]),
                           p_value=float(r["p_value"])))
    return pooled


# ----------------------------------------------------------------------------
# Loading the real inputs (needs the lab data)
# ----------------------------------------------------------------------------
def _region_array(fr_db, region, session, stim, response_range):
    """Primary-style loader with the Dorsal/Posterior fallback of CCA_null_dist_test.py."""
    try:
        return fr_db.get_firing_rate_array(region, session, stim, response_range,
                                           NEURON_THRESHOLD, CCA_RANDOM_STATE)
    except UnboundLocalError:
        if region != "Dorsal auditory area":
            raise
        return fr_db.get_firing_rate_array("Posterior auditory area", session, stim, response_range,
                                           NEURON_THRESHOLD, CCA_RANDOM_STATE)


def condition_rows(feather_df):
    c = CONDITION
    rows = feather_df[(feather_df["source_region"] == c["source"])
                      & (feather_df["target1"] == c["target1"]) & (feather_df["target2"] == c["target2"])
                      & (feather_df["stimulus"] == c["stimulus"])
                      & (feather_df["response_range"] == c["response_range"])]
    if rows.empty:
        raise RuntimeError(f"No rows in the feather file for {c}")
    return rows.sort_values("session")


def build_example(data_dir, fr_db=None, feather_path=None):
    import pandas as pd
    if fr_db is None:
        from analysis_class import FiringRateAnalysis
        fr_db = FiringRateAnalysis(db_suffix="coords_updated")
    feather_path = feather_path or f"{fp.FIGDATA_ROOT}/CCA_cross_region_projection_similarity.feather"
    rows = condition_rows(pd.read_feather(feather_path))
    print(f"[fig5] {len(rows)} session(s) for {CONDITION}")

    row = rows[rows["session"] == EXAMPLE_SESSION].iloc[0] if EXAMPLE_SESSION else rows.iloc[0]
    c = CONDITION
    X = _region_array(fr_db, c["source"], row["session"], c["stimulus"], c["response_range"])
    Y1 = _region_array(fr_db, c["target1"], row["session"], c["stimulus"], c["response_range"])
    Y2 = _region_array(fr_db, c["target2"], row["session"], c["stimulus"], c["response_range"])
    n_trials = min(len(X), len(Y1), len(Y2))
    X, Y1, Y2 = X[:n_trials], Y1[:n_trials], Y2[:n_trials]

    example = compute_example(X, Y1, Y2, int(row["pair1_components"]), int(row["pair2_components"]))
    example["session"] = str(row["session"])
    example["R_actual_stored"] = float(row["R_actual"])
    ok = np.isclose(example["R_actual"], example["R_actual_stored"], atol=1e-6)
    print(f"[fig5] example session {example['session']}: recomputed R = {example['R_actual']:.6f}, "
          f"stored R = {example['R_actual_stored']:.6f} -> {'MATCH' if ok else 'MISMATCH (inputs differ from the original run)'}")
    print(f"[fig5] R with orthonormalized U, V = {example['R_actual_orthonormalized']:.6f} "
          f"(vs {example['R_actual']:.6f} as computed in the pipeline)")

    row_dict = row.to_dict()
    example["null_R"] = np.asarray(row_dict["null_similarities"], dtype=np.float32)
    example["null_mean"], example["null_std"] = float(row["null_mean"]), float(row["null_std"])
    example["z_actual"], example["p_value"] = float(row["z_score_diff_from_null"]), float(row["p_value"])

    os.makedirs(data_dir, exist_ok=True)
    pickle.dump(dict(condition=CONDITION, example=example,
                     pooled=pooled_from_rows([r.to_dict() for _, r in rows.iterrows()])),
                open(f"{data_dir}/null_example.pkl", "wb"))
    print(f"[fig5] saved {data_dir}/null_example.pkl")


def main(data_dir=None, **kwargs):
    build_example(data_dir or fp.figure_data_dir(FIG), **kwargs)


if __name__ == "__main__":
    main()
