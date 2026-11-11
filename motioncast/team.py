"""Unsupervised team assignment: SigLIP crop embeddings -> UMAP -> KMeans(2).

Embedding approach from Roboflow `sports` (MIT). Changes here: teams are assigned
once per *track* by majority vote over sampled crops (stable colors, one model call
per crop instead of per frame), and goalkeepers are assigned by which team's
defensive half they stand in.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

SIGLIP_MODEL = "google/siglip-base-patch16-224"


def center_crop(crop: np.ndarray) -> np.ndarray:
    """Keep the torso: drop the head and legs and the box edges (less grass, more shirt)."""
    h, w = crop.shape[:2]
    return crop[int(h * 0.15): int(h * 0.6), int(w * 0.15): int(w * 0.85)]


class CropEmbedder:
    def __init__(self, device: str = "cpu", batch_size: int = 32):
        import torch
        from transformers import SiglipImageProcessor, SiglipVisionModel

        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        self.model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL).to(device).eval()
        self.processor = SiglipImageProcessor.from_pretrained(SIGLIP_MODEL)

    def __call__(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        from PIL import Image

        images = [Image.fromarray(c[:, :, ::-1]) for c in crops_bgr]
        out = []
        with self.torch.no_grad():
            for i in range(0, len(images), self.batch_size):
                inputs = self.processor(images=images[i:i + self.batch_size], return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                hidden = self.model(**inputs).last_hidden_state
                out.append(hidden.mean(dim=1).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 768), dtype=np.float32)


def cluster_two_teams(features: np.ndarray, seed: int = 0) -> np.ndarray:
    """Cluster embeddings into 2 groups. Returns labels in {0, 1}."""
    from sklearn.cluster import KMeans

    if len(features) < 2:
        return np.zeros(len(features), dtype=int)
    if len(features) >= 30:
        import warnings

        import umap

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            reduced = umap.UMAP(n_components=3, random_state=seed, n_jobs=1).fit_transform(features)
    else:
        from sklearn.decomposition import PCA

        reduced = PCA(n_components=min(3, len(features) - 1), random_state=seed).fit_transform(features)
    return KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(reduced)


def vote_track_teams(crop_track_ids: list[int], crop_labels: np.ndarray) -> dict[int, int]:
    """Majority vote of crop cluster labels per track."""
    votes: dict[int, Counter] = defaultdict(Counter)
    for tid, label in zip(crop_track_ids, crop_labels):
        votes[int(tid)][int(label)] += 1
    return {tid: c.most_common(1)[0][0] for tid, c in votes.items()}


def assign_goalkeepers(
    gk_mean_x: dict[int, float],
    team_mean_x: dict[int, float],
) -> dict[int, int]:
    """A goalkeeper belongs to the team whose average position is nearer their goal.

    gk_mean_x: track id -> mean pitch x (cm) of the goalkeeper.
    team_mean_x: team id -> mean pitch x (cm) of that team's outfield players.
    """
    if len(team_mean_x) < 2:
        return {tid: next(iter(team_mean_x), 0) for tid in gk_mean_x}
    out = {}
    for tid, x in gk_mean_x.items():
        out[tid] = min(team_mean_x, key=lambda t: abs(team_mean_x[t] - x))
    return out
