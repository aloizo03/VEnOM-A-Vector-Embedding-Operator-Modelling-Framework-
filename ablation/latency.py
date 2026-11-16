import argparse
import os
import time
import types

import numpy as np
import pandas as pd
import torch

from ablation import config as A
from ablation.models import build_model
from ablation.pipeline import TableSet, list_csvs, load_encoder, max_cells
from utils_.select_datasets import Data_selection
from utils_.utils import set_global_seed

HERE = os.path.dirname(os.path.abspath(__file__))

ENCODERS = {"NumTab2Vec": None, "Num2Vec (retrained)": "Num2Vec", "ae": "Num2VecAE",
            "siamese": "Num2VecSiamese", "mlp_vae": "MLPEncoderVAE", "mlp_ae": "MLPEncoderAE"}

VARIANT_ROW = {"vae": "Num2Vec (retrained)", **{v: "Num2Vec (retrained)" for v in A.VARIANTS
                                                 if v.startswith("beta_")},
               "ae": "ae", "siamese": "siamese", "mlp_vae": "mlp_vae", "mlp_ae": "mlp_ae"}


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def vectorise(model, mi, files, dropout, dev):
    model.train(dropout)
    cap = max_cells(mi)
    vecs, rows = {}, []
    with torch.no_grad():
        for f in files:
            t0 = time.perf_counter()
            dset = TableSet([f], mi["dim_k"], cap)
            t1 = time.perf_counter()
            x = torch.tensor(dset[0][None], dtype=torch.float64, device=dev)
            sync(dev)
            t2 = time.perf_counter()
            _, mu, _ = model(x)
            v = mu[0].cpu().numpy().astype(np.float64)
            sync(dev)
            t3 = time.perf_counter()
            vecs[f] = v
            rows.append(dict(file=f, read_s=t1 - t0, prep_s=t2 - t1, fwd_s=t3 - t2, total_s=t3 - t0))
    return vecs, rows


def select_times(lake_vecs, query_vecs):
    ns = types.SimpleNamespace(vectors=lake_vecs)
    out = []
    for q, v in query_vecs.items():
        t0 = time.perf_counter()
        dist = Data_selection.compute_distance(ns, vect_dict=lake_vecs, vectors={q: v})
        Data_selection.select_most_relevants_data(ns, dist, {q: v}, ratio=A.LAMBDA, compare="desc")
        out.append(dict(file=q, select_s=time.perf_counter() - t0))
    return out


def median_rows(runs):
    return [dict(file=rs[0]["file"], **{k: float(np.median([r[k] for r in rs])) for k in rs[0] if k != "file"})
            for rs in zip(*runs)]


def run_dim(dim, datasets, encoders, warmup, max_files, out_dir, repeats):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()
    published, mi = load_encoder(A.RC.NUM2VEC_CKPT[dim])
    load_s = time.perf_counter() - t0
    print(f"d={dim}: device {dev}"
          f"{' (' + torch.cuda.get_device_name(0) + ')' if dev.type == 'cuda' else ''}, "
          f"published checkpoint loaded in {load_s:.1f} s", flush=True)
    files = {}
    for ds in datasets:
        lake, queries = list_csvs(A.DATASETS[ds]["lake"]), list_csvs(A.DATASETS[ds]["queries"])
        files[ds] = (lake[:max_files], queries[:max_files]) if max_files else (lake, queries)
        for f in files[ds][0] + files[ds][1]:          # warm the file cache, not timed
            pd.read_csv(f)
    raw = []
    for enc in encoders:
        set_global_seed(0)
        if ENCODERS[enc] is None:
            model = published
        else:
            model = build_model(ENCODERS[enc], mi, train=False)
        n_params = sum(p.numel() for p in model.parameters())
        for ds in datasets:
            lake, queries = files[ds]
            vectorise(model, mi, lake[:warmup], True, dev)          # warm-up, not recorded
            L, lr = vectorise(model, mi, lake, True, dev)
            runs = [vectorise(model, mi, queries, False, dev) for _ in range(repeats)]
            Q, qr = runs[0][0], median_rows([r for _, r in runs])
            sr = {r["file"]: r["select_s"] for r in median_rows([select_times(L, Q) for _ in range(repeats)])}
            for part, rows in (("lake", lr), ("query", qr)):
                for r in rows:
                    raw.append(dict(dim=dim, encoder=enc, dataset=ds, part=part, n_params=n_params,
                                    select_s=sr.get(r["file"]) if part == "query" else None, **r))
            print(f"  {enc:20s} {ds:8s} lake {len(lake):4d} tables "
                  f"{sum(r['total_s'] for r in lr):8.2f} s | query vectorise "
                  f"{1e3 * np.mean([r['total_s'] for r in qr]):8.1f} ms, select "
                  f"{1e3 * np.mean(list(sr.values())):6.1f} ms", flush=True)
        if model is not published:
            del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    del published
    raw = pd.DataFrame(raw)
    raw.to_csv(os.path.join(out_dir, f"latency_raw_d{dim}.csv"), index=False)

    q = raw[raw.part == "query"].groupby(["dim", "encoder", "dataset"])
    lk = raw[raw.part == "lake"].groupby(["dim", "encoder", "dataset"])
    summ = pd.DataFrame({
        "n_params": q.n_params.first(),
        "n_lake": lk.size(), "n_queries": q.size(),
        "lake_vectorise_total_s": lk.total_s.sum(),
        "lake_vectorise_per_table_ms": 1e3 * lk.total_s.mean(),
        "lake_vectorise_per_table_median_ms": 1e3 * lk.total_s.median(),
        "lake_vectorise_robust_s": lk.total_s.median() * lk.size(),
        "query_read_ms": 1e3 * q.read_s.mean(), "query_prep_ms": 1e3 * q.prep_s.mean(),
        "query_fwd_ms": 1e3 * q.fwd_s.mean(),
        "query_vectorise_ms": 1e3 * q.total_s.mean(), "query_vectorise_median_ms": 1e3 * q.total_s.median(), "query_vectorise_std_ms": 1e3 * q.total_s.std(),
        "select_ms": 1e3 * q.select_s.mean(), "select_std_ms": 1e3 * q.select_s.std(),
    }).reset_index()
    summ["published_ckpt_load_s"] = load_s
    summ["device"] = torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"
    summ.to_csv(os.path.join(out_dir, f"latency_d{dim}.csv"), index=False)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="+", default=[100, 200, 300])
    ap.add_argument("--datasets", nargs="+", default=list(A.DATASETS))
    ap.add_argument("--encoders", nargs="+", default=list(ENCODERS), choices=list(ENCODERS))
    ap.add_argument("--warmup", type=int, default=5, help="lake tables vectorised first, not timed")
    ap.add_argument("--repeats", type=int, default=3, help="query repeats; the median is kept")
    ap.add_argument("--max-files", type=int, default=None, help="testing only: limit lake/query files")
    ap.add_argument("--out", default=os.path.join(HERE, "latency"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if torch.cuda.is_available():
        print("check with nvidia-smi that no other job uses this GPU: shared GPUs distort the timings")
    for dim in a.dims:
        run_dim(dim, a.datasets, a.encoders, a.warmup, a.max_files, a.out, a.repeats)
    print("written to", a.out)


if __name__ == "__main__":
    main()
