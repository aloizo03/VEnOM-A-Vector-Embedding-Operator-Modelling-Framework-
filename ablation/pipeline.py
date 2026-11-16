import glob
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import yaml

from data.Dataset import DataBuilder_Pipeline
from models.Loss import Loss_VAE
from models.utils import get_optimiser, get_parameter_names
from utils_.utils import set_global_seed
from ablation.models import Loss_AE, Loss_BetaVAE, Loss_NTXent, build_model


def max_cells(model_info):
    return (model_info["d_numerical"] * model_info["dim_k"]) // 2


def list_csvs(path):
    return sorted(glob.glob(os.path.join(path, "*.csv"))) if os.path.isdir(path) else [path]


def training_files(data_config, repo, per_dataset=False):
    with open(data_config) as fh:
        conf = yaml.safe_load(fh)
    files, counts = [], {}
    for name, d in conf.items():
        p = d["path"] if os.path.isabs(d["path"]) else os.path.join(repo, d["path"])
        found = sorted(glob.glob(os.path.join(p, "*.csv"))) if os.path.isdir(p) else []
        if not found:
            raise FileNotFoundError(f"training data of '{name}' not found: {p} (from {data_config})")
        counts[name] = len(found)
        files.extend(found)
    return (files, counts) if per_dataset else files


class TableSet(torch.utils.data.Dataset):

    def __init__(self, files, dim_k, cap):
        self.builder = DataBuilder_Pipeline(datasets_filepaths=list(files), norm=True, ret_Table=True,
                                            ret_label=True, k_dim=dim_k, ret_all_dataset=True)
        self.files = list(files)
        self.dim_k = dim_k
        self.cap = cap
        self.truncated = set()

    def __len__(self):
        return len(self.files)

    def _full(self, idx):
        x, _ = self.builder[idx]
        if x.shape[0] > self.cap:
            self.truncated.add(self.files[idx])
            x = x[:self.cap]
        return x

    def view(self, idx, rng, frac=(0.5, 0.9)):
        n_cols = self.builder.datasets_data[idx].shape[1]
        n_rows = self.builder.datasets_data[idx].shape[0]
        table = self.builder[idx][0].reshape(-1, n_cols)[:n_rows]
        keep = max(1, int(round(n_rows * rng.uniform(*frac))))
        rows = np.sort(rng.choice(n_rows, size=keep, replace=False))
        out = np.zeros((max(self.dim_k, keep), n_cols))
        out[:keep] = table[rows]
        return out.flatten()[:self.cap]

    def __getitem__(self, idx):
        return self._full(idx)


def pad_batch(arrays):
    n = max(a.shape[0] for a in arrays)
    out = np.zeros((len(arrays), n))
    for i, a in enumerate(arrays):
        out[i, :a.shape[0]] = a
    return torch.tensor(out, dtype=torch.float64)


def make_loss(objective, beta):
    if objective == "vae":
        return Loss_VAE() if float(beta) == 1.0 else Loss_BetaVAE(beta=beta)
    if objective == "ae":
        return Loss_AE()
    if objective == "siamese":
        return Loss_NTXent()
    raise ValueError(objective)


class AblationTrainer:

    def __init__(self, class_name, objective, beta, model_info, train_files, out_dir, epochs=100,
                 batch_size=4, lr=5e-4, optimiser="Adam", weight_decay=0.1, momentum=0.8, lrf=0.1,
                 val_frac=0.1, seed=0, max_hours=None, micro_batch=None):
        set_global_seed(seed)
        self.seed, self.class_name, self.objective, self.beta = seed, class_name, objective, beta
        self.model_info = dict(model_info)
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.epochs, self.batch_size, self.max_hours = epochs, batch_size, max_hours
        self.micro = min(batch_size, micro_batch or batch_size)
        self.rng = np.random.default_rng(seed)

        files = list(train_files)
        random.Random(seed).shuffle(files)
        n_val = max(1, int(round(len(files) * val_frac)))
        cap = max_cells(model_info)
        self.train_set = TableSet(files[n_val:], model_info["dim_k"], cap)
        self.val_set = TableSet(files[:n_val], model_info["dim_k"], cap)

        self.model = build_model(class_name, model_info, train=(objective != "siamese"))
        self.device = next(self.model.parameters()).device
        self.loss_fn = make_loss(objective, beta)

        names = get_parameter_names(self.model, [nn.LayerNorm])
        decay = [n for n in names if "bias" not in n]
        groups = [{"params": [p for n, p in self.model.named_parameters() if n in decay], "lr": lr,
                   "weight_decay": weight_decay},
                  {"params": [p for n, p in self.model.named_parameters() if n not in decay], "lr": lr,
                   "weight_decay": 0}]
        self.opt = get_optimiser(optimiser_name=optimiser, parameters=groups, lr=lr,
                                 weight_decay=weight_decay, momentum=momentum)
        lf = lambda x: (1 - x / max(1, epochs - 1)) * (1.0 - lrf) + lrf
        self.sched = lr_scheduler.LambdaLR(self.opt, lr_lambda=lf)
        self.hparams = dict(epochs=epochs, batch_size=batch_size, micro_batch=self.micro, lr=lr, optimiser=optimiser,
                            weight_decay=weight_decay, lrf=lrf, val_frac=val_frac, seed=seed)

    # reconstruction objectives: (rec, kl) of a micro-batch of dataset indices
    def _terms(self, dset, idxs):
        x = pad_batch([dset[i] for i in idxs]).to(self.device)
        recon, mu, s = self.model(x)
        recon = recon[:, :x.shape[1]]
        return self.loss_fn(recon, x, mu, s)

    def _chunks(self, idxs):
        return [idxs[i:i + self.micro] for i in range(0, len(idxs), self.micro)]

    def _embed_views(self, views, grad):
        zs, states = [], []
        for c in [views[i:i + self.micro] for i in range(0, len(views), self.micro)]:
            states.append((torch.get_rng_state(),
                           torch.cuda.get_rng_state() if torch.cuda.is_available() else None))
            with torch.set_grad_enabled(grad):
                zs.append(self.model(pad_batch(c).to(self.device))[1])
        return torch.cat(zs), states

    def _accumulate(self, dset, idxs, backward=True):
        if self.objective == "siamese":
            views = [dset.view(i, self.rng) for i in idxs] + [dset.view(i, self.rng) for i in idxs]
            z, states = self._embed_views(views, grad=False)
            z = z.detach().requires_grad_(backward)
            n = len(idxs)
            with torch.set_grad_enabled(backward):
                loss = self.loss_fn(z[:n], z[n:])
            if backward:
                loss.backward()
                g = z.grad
                k = 0
                for c, (cpu_s, cuda_s) in zip([views[i:i + self.micro] for i in range(0, len(views), self.micro)],
                                              states):
                    torch.set_rng_state(cpu_s)
                    if cuda_s is not None:
                        torch.cuda.set_rng_state(cuda_s)
                    zc = self.model(pad_batch(c).to(self.device))[1]
                    (zc * g[k:k + len(c)]).sum().backward()
                    k += len(c)
            return loss.item(), loss.item(), 0.0
        tot = rec = kl = 0.0
        for c in self._chunks(idxs):
            with torch.set_grad_enabled(backward):
                r, k = self._terms(dset, c)
                part = r * (len(c) / len(idxs)) + k
            if backward:
                part.backward()
            tot, rec, kl = tot + part.item(), rec + r.item() * len(c) / len(idxs), kl + k.item()
        return tot, rec, kl

    def _batches(self, n, shuffle):
        order = self.rng.permutation(n) if shuffle else np.arange(n)
        for i in range(0, n, self.batch_size):
            yield order[i:i + self.batch_size].tolist()

    def _evaluate(self, dset):
        self.model.eval()
        tot = rec = kl = 0.0
        nb = 0
        with torch.no_grad():
            for b in self._batches(len(dset), shuffle=False):
                l, r, k = self._accumulate(dset, b, backward=False)
                tot, rec, kl, nb = tot + l, rec + r, kl + k, nb + 1
        return tot / nb, rec / nb, kl / nb

    def train(self):
        best, log, t0 = math.inf, [], time.time()
        ckpt = os.path.join(self.out_dir, "best.pt")
        for ep in range(self.epochs):
            self.model.train()
            tot = nb = 0
            for b in self._batches(len(self.train_set), shuffle=True):
                self.opt.zero_grad()
                loss, _, _ = self._accumulate(self.train_set, b)
                self.opt.step()
                tot, nb = tot + loss, nb + 1
            self.sched.step()
            v, vr, vk = self._evaluate(self.val_set)
            log.append(dict(epoch=ep, train_loss=tot / nb, val_loss=v, val_rec=vr, val_kl=vk,
                            elapsed_s=time.time() - t0))
            print(json.dumps(log[-1]), flush=True)
            if v < best:
                best = v
                self.save(ckpt, ep, best, log)
            if self.max_hours and time.time() - t0 > self.max_hours * 3600:
                print(f"stopping: time budget of {self.max_hours} h reached", flush=True)
                break
        self.save(os.path.join(self.out_dir, "last.pt"), ep, best, log)
        with open(os.path.join(self.out_dir, "train_log.json"), "w") as fh:
            json.dump(dict(log=log, truncated=sorted(self.train_set.truncated | self.val_set.truncated),
                           train_time_s=time.time() - t0, n_params=sum(p.numel() for p in self.model.parameters()),
                           **self.hparams), fh, indent=1)
        return ckpt

    def save(self, path, ep, best, log):
        torch.save({"class_name": self.class_name, "objective": self.objective, "beta": self.beta,
                    "model_info": self.model_info, "model_state_dict": self.model.state_dict(),
                    "epoch": ep, "best_loss": best, "log": log, "hparams": self.hparams}, path)


def load_encoder(ckpt_path, class_name=None):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    name = class_name or ck.get("class_name", "Num2Vec")
    model = build_model(name, ck["model_info"], train=False)
    model.load_state_dict(ck["model_state_dict"])
    return model, ck["model_info"]


def untrained_encoder(class_name, model_info, seed):
    set_global_seed(seed)
    return build_model(class_name, model_info, train=False)


def embed_files(model, model_info, files, eval_mode=True):
    model.train(not eval_mode)
    dset = TableSet(files, model_info["dim_k"], max_cells(model_info))
    out = {}
    with torch.no_grad():
        for i, f in enumerate(dset.files):
            x = torch.tensor(dset[i][None], dtype=torch.float64, device=next(model.parameters()).device)
            _, mu, _ = model(x)
            out[f] = mu[0].detach().cpu().numpy().astype(np.float64)
    return out, sorted(dset.truncated)


def latent_stats(model, model_info, files):
    model.eval()
    dset = TableSet(files, model_info["dim_k"], max_cells(model_info))
    mus, ss = [], []
    with torch.no_grad():
        for i in range(len(dset)):
            x = torch.tensor(dset[i][None], dtype=torch.float64, device=next(model.parameters()).device)
            _, mu, s = model(x)
            mus.append(mu[0].cpu().numpy())
            ss.append(s[0].cpu().numpy())
    mu, s = np.array(mus), np.array(ss)
    kl = -0.5 * (1 + s - mu ** 2 - np.exp(s))
    d = np.linalg.norm(mu[:, None] - mu[None], axis=-1)[np.triu_indices(len(mu), 1)]
    return dict(n=len(mu), dim=mu.shape[1], active_units=int((mu.var(0) > 0.01).sum()),
                kl_per_dim=float(kl.mean()), mu_var_mean=float(mu.var(0).mean()),
                pairwise_l2_median=float(np.median(d)) if len(d) else float("nan"),
                pairwise_l2_min=float(d.min()) if len(d) else float("nan"))
