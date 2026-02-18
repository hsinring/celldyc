from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from anndata import AnnData
from scipy import sparse
from torch.utils.data import TensorDataset, DataLoader
import random

from celldyc.tools.utils import make_label


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class Clock(nn.Module):

    def __init__(self, n_in: int):
        super().__init__()
        self.emb = nn.Linear(n_in, 1)

    def forward(self, x):
        z = self.emb(x)
        return z


class Embedding(nn.Module):

    def __init__(self, n_in: int, n_lat: int):
        super().__init__()
        self.emb = nn.Linear(n_in, n_lat)

    def forward(self, x):
        z = self.emb(x)
        return z


class Decoder(nn.Module):

    def __init__(self, n_lat: int, n_genes: int):
        super().__init__()
        self.n_genes = n_genes
        self.net = nn.Sequential(
            nn.Linear(n_lat, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()
        )
        self.trend = nn.Sequential(nn.Linear(128, n_genes), nn.Sigmoid())
        self.change_prob = nn.Sequential(nn.Linear(128, n_genes), nn.Sigmoid())

    def forward(self, z):
        h = self.net(z)
        trend = self.trend(h) - 0.5
        prob = self.change_prob(h)

        return trend, prob


class semi_model(nn.Module):

    def __init__(self, n_genes: int, n_lat: int = 10):
        super().__init__()

        self.clock = Clock(n_genes)
        self.encoder = Embedding(n_genes, n_lat - 1)
        self.decoder = Decoder(n_lat, n_genes)

    def forward(self, x, return_z=False):
        c = self.clock(x)
        g = self.encoder(x)
        z = torch.cat((c, g), dim=1)
        trend, prob = self.decoder(z)

        if return_z:
            return z
        else:
            return trend * prob, c


class EarlyStop:
    def __init__(self, patience: int = 250, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = float("inf")

    def __call__(self, loss: float) -> bool:
        if loss < self.best - self.min_delta:
            self.best = loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


class Trainer:
    def __init__(
        self,
        model: semi_model,
        lr: float = 1e-2,
        wd: float = 1e-2,
        patience: int = 20,
        min_delta: float = 1e-4,
        time_weight=0.1,
    ):
        self.model = model
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        self.early_stop = EarlyStop(patience=patience, min_delta=min_delta)
        self.time_weight = time_weight

    def train_epoch(self, X, trend_label, reliability, time):
        self.opt.zero_grad()
        trend_pred, clock = self.model(X)

        cosine = (1 - F.cosine_similarity(trend_pred, trend_label, dim=1)) * reliability
        trend_loss = cosine.mean() / reliability.mean()

        time = time.reshape(-1, 1)
        time_pairwise = time - time.T
        time_loss = (
            1
            - F.cosine_similarity(
                (clock - clock.T) * torch.sign(torch.abs(time_pairwise)),
                time_pairwise,
                dim=1,
            )
        ).mean()

        loss = trend_loss + self.time_weight * time_loss
        loss.backward()
        self.opt.step()

        return {
            "loss": loss.item(),
            "trend_loss": trend_loss.item(),
            "time_loss": time_loss.item(),
        }


class Model:
    def __init__(
        self,
        adata: AnnData,
        n_latent: int = 10,
        input_layer=None,
        label_layer="raw_label",
        reliability_layer="reliability",
        time_key="day",
        seed=42,
        model_path=None,
        dataloader_with_seed=False,
        **kw,
    ):
        set_seed(seed=seed)
        self.seed = seed
        self.loader_reproduce = dataloader_with_seed

        self.n_genes = adata.n_vars
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_path is None:
            self.model = semi_model(self.n_genes, n_latent, **kw).to(self.device)
        else:
            self.model = torch.load(
                model_path, map_location=self.device, weights_only=False
            )

        adata_label = make_label(adata, time_key=time_key)
        label = adata_label.layers[label_layer].copy()

        if sparse.issparse(label):
            label = label.toarray()

        R = adata_label.obs[reliability_layer].to_numpy()
        T = adata_label.obs["time_num"].to_numpy()

        if input_layer is None:
            X = adata.X.copy()
        else:
            X = adata.layers[input_layer].copy()
        if sparse.issparse(X):
            X = X.toarray()

        self.X = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.label = torch.tensor(label, dtype=torch.float32, device=self.device)
        self.R = torch.tensor(R, dtype=torch.float32, device=self.device)
        self.T = torch.tensor(T, dtype=torch.float32, device=self.device)

        self.connect = adata.obsp["connectivities"].copy().tocoo()
        self.connect = (
            torch.sparse_coo_tensor(
                np.vstack([self.connect.row, self.connect.col]),
                self.connect.data,
                size=self.connect.shape,
            )
            .coalesce()
            .to(self.device)
        )

    def train(
        self,
        n_epochs: int = None,
        max_epochs: int = 500,
        batch_size: int = None,
        lr: float = 1e-2,
        patience: int = 40,
        min_delta: float = 1e-6,
        time_weight=0.1,
    ):
        trainer = Trainer(
            self.model,
            lr=lr,
            patience=patience,
            min_delta=min_delta,
            time_weight=time_weight,
        )

        dataset = TensorDataset(self.X, self.label, self.R, self.T)
        batch_size = batch_size or min(256, len(dataset))

        if self.loader_reproduce:
            generator = torch.Generator()
            generator.manual_seed(self.seed)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                generator=generator,
            )
        else:
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True, drop_last=False
            )

        if n_epochs is not None:
            total_epochs = n_epochs
            use_early_stop = False
            print(f"Training for exactly {n_epochs} epochs (early stop disabled)")
        else:
            total_epochs = max_epochs
            use_early_stop = True
            print(
                f"Training with early stop (max_epochs={max_epochs}, patience={patience})"
            )

        for epoch in range(total_epochs):
            epoch_loss, epoch_trend, epoch_time = [], [], []
            for x, l_, r, t in loader:
                log = trainer.train_epoch(x, l_, r, t)
                epoch_loss.append(log["loss"])
                epoch_trend.append(log["trend_loss"])
                epoch_time.append(log["time_loss"])

            avg_loss = sum(epoch_loss) / len(epoch_loss)
            avg_trend = sum(epoch_trend) / len(epoch_trend)
            avg_time = sum(epoch_time) / len(epoch_time)

            if epoch % 50 == 0 or epoch == total_epochs - 1:
                print(
                    f"epoch {epoch + 1:3d}:"
                    f"loss={avg_loss:.6f},"
                    f"trend_loss={avg_trend:.6f},"
                    f"time_loss={avg_time:.6f}"
                )

            if use_early_stop and trainer.early_stop(avg_loss):
                print(f"Early stopping at epoch {epoch}")
                break

    def get_trends(self, batch_size=None, prob_filter=0.05) -> pd.DataFrame:
        self.model.eval()
        X_all = self.X
        N, _ = X_all.shape
        batch_size = batch_size or min(512, N)

        split_trends = []
        split_probs = []

        with torch.no_grad():
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                X = X_all[start:end]
                z = self.model(X, return_z=True)
                trend, prob = self.model.decoder(z)

                split_trends.append(trend.cpu())
                split_probs.append(prob.cpu())

        trends = torch.cat(split_trends, dim=0).numpy()
        probs = torch.cat(split_probs, dim=0).numpy()

        trends = np.sign(trends)
        probs = np.where(probs > prob_filter, 1, 0)

        return trends * probs

    def get_time(self, batch_size=None) -> pd.DataFrame:
        self.model.eval()
        X_all = self.X
        N, _ = X_all.shape
        batch_size = batch_size or min(512, N)

        split_time = []
        with torch.no_grad():
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                X = X_all[start:end]
                _, clock = self.model(X)
                split_time.append(clock.cpu())

        time = torch.cat(split_time, dim=0).numpy()
        time = (time - time.min()) / (time.max() - time.min())

        return time

    def clock_weights(self):
        w = self.model.clock.emb.weight.detach().cpu().numpy().squeeze()
        w = w / np.abs(w).max()

        return w

    def save_model(self, path: str):
        self.model.eval()
        torch.save(self.model, path)
