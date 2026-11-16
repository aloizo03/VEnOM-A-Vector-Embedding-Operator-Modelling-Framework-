import argparse
import csv
import json
import os
import pickle
import platform
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "rerun"))

from ablation import config as A
import run_all as RA

PY = sys.executable
STAGES = ["setup", "train", "embed", "select", "opmodel", "table"]


def variant_seeds(name):
    v = A.VARIANTS[name]
    return [0] if "ckpt" in v else A.TRAIN_SEEDS


def modes_for(name):
    return A.EMBED_MODES


def all_runs(datasets):
    for name in A.VARIANTS:
        for s in variant_seeds(name):
            for ds in datasets:
                for mode in modes_for(name):
                    yield name, s, ds, mode


def model_info_from(ckpt_path):
    import torch
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ck["model_info"], ck.get("data_config")


def cached_model_info(R):
    p = R.file("model_info.json")
    if not os.path.isfile(p):
        mi, dc = model_info_from(A.PAPER_CKPT)
        if A.SMOKE:
            mi = {**mi, **A.SMOKE["model_info"]}
        json.dump({"model_info": mi, "data_config": dc}, open(p, "w"), indent=1)
    d = json.load(open(p))
    return d["model_info"], d["data_config"]


def train_data_config(dc=None):
    for cand in (A.TRAIN_DATA_CONFIG, dc):
        if cand and os.path.isfile(cand if os.path.isabs(cand) else os.path.join(REPO, cand)):
            return cand if os.path.isabs(cand) else os.path.join(REPO, cand)
    raise FileNotFoundError("no training data config found")


def worker_train(a):
    from ablation.pipeline import AblationTrainer, training_files
    v = A.VARIANTS[a.variant]
    info = json.load(open(a.model_info))
    files, counts = training_files(train_data_config(info["data_config"]), REPO, per_dataset=True)
    print(f"training on all datasets: {counts}", flush=True)
    json.dump(counts, open(os.path.join(a.out, "train_datasets.json"), "w"), indent=1)
    if A.SMOKE:
        files = files[::max(1, len(files) // A.SMOKE["train_files"])][:A.SMOKE["train_files"]]
    print(f"{a.variant}: {v['cls']} objective={v['objective']} beta={v.get('beta')} seed={a.seed} "
          f"train files={len(files)}", flush=True)
    t = AblationTrainer(v["cls"], v["objective"], v.get("beta", 1.0), info["model_info"], files, a.out,
                        seed=a.seed, max_hours=A.MAX_HOURS_PER_MODEL, **A.TRAIN)
    t.train()


def worker_embed(a):
    import torch
    from ablation.pipeline import embed_files, list_csvs, load_encoder
    from utils_.utils import set_global_seed
    v = A.VARIANTS[a.variant]
    cfg = A.DATASETS[a.dataset]
    lake, queries = list_csvs(cfg["lake"]), list_csvs(cfg["queries"])
    if A.SMOKE:
        lake, queries = lake[:A.SMOKE["lake"]], queries[:A.SMOKE["queries"]]
    info = json.load(open(a.model_info))
    t0 = time.time()
    if "ckpt" in v:
        model, mi = load_encoder(v["ckpt"], class_name=v["cls"])
    else:
        model, mi = load_encoder(a.ckpt)
    set_global_seed(a.seed)
    # paper mode: lake as Vectorise (dropout on), queries as Data_selection (dropout off)
    L, t1 = embed_files(model, mi, lake, eval_mode=(a.mode == "eval"))
    Q, t2 = embed_files(model, mi, queries, eval_mode=True)
    trunc = sorted(set(t1) | set(t2))
    del model
    torch.cuda.empty_cache()
    pickle.dump({"lake": L, "queries": Q, "truncated": trunc,
                 "embed_time_s": time.time() - t0}, open(a.out, "wb"))


def worker(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=["_train", "_embed"])
    ap.add_argument("--variant"); ap.add_argument("--seed", type=int); ap.add_argument("--dataset")
    ap.add_argument("--mode"); ap.add_argument("--ckpt"); ap.add_argument("--out")
    ap.add_argument("--model-info")
    a = ap.parse_args(argv)
    {"_train": worker_train, "_embed": worker_embed}[a.job](a)


def stage_setup(R, datasets):
    import torch
    env = {"time": time.strftime("%F %T"), "host": platform.node(), "python": sys.version,
           "torch": torch.__version__, "cuda": torch.cuda.is_available(),
           "gpu": RA.sh("nvidia-smi --query-gpu=name,memory.total --format=csv"),
           "git_commit": RA.sh(f"git -c safe.directory={REPO} -C {REPO} rev-parse HEAD"),
           "git_branch": RA.sh(f"git -c safe.directory={REPO} -C {REPO} branch --show-current"),
           "dim": A.DIM, "paper_ckpt": A.PAPER_CKPT, "train": A.TRAIN, "train_seeds": A.TRAIN_SEEDS,
           "operator_seeds": A.SEEDS, "lambda": A.LAMBDA, "embed_modes": A.EMBED_MODES}
    ok = os.path.isfile(A.PAPER_CKPT)
    if ok:
        mi, dc = cached_model_info(R)
        env["model_info"], env["train_data_config"] = mi, train_data_config(dc)
    try:
        from ablation.pipeline import training_files
        env["train_files_per_dataset"] = training_files(train_data_config(), REPO, per_dataset=True)[1]
    except FileNotFoundError as e:
        env["train_files_per_dataset"] = f"MISSING: {e}"
        ok = False
    for ds in datasets:
        for side in ("lake", "queries"):
            p = A.DATASETS[ds][side]
            exists = os.path.isdir(p)
            ok &= exists
            env[f"{ds}_{side}"] = f"{p} ({'ok' if exists else 'MISSING'})"
    json.dump(env, open(R.file("env.json"), "w"), indent=1, default=str)
    with open(R.file("plan.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "seed", "dataset", "mode", "operators"])
        for name, s, ds, mode in all_runs(datasets):
            w.writerow([name, s, ds, mode, " ".join(A.DATASETS[ds]["operators"])])
    print(json.dumps({k: env[k] for k in env if k != "model_info"}, indent=1, default=str))
    return ok


def ckpt_of(R, name, seed):
    return os.path.join(R.out, "train", f"{name}_s{seed}", "best.pt")


def stage_train(R, datasets):
    info = R.file("model_info_full.json")
    if not os.path.isfile(info):
        mi, dc = cached_model_info(R)
        json.dump({"model_info": mi, "data_config": dc}, open(info, "w"))
    for name, v in A.VARIANTS.items():
        if "objective" not in v:
            continue
        for s in variant_seeds(name):
            d = R.dir("train", f"{name}_s{s}")
            R.run("train", f"train_{name}_s{s}",
                  [PY, "-m", "ablation.run_ablation", "_train", "--variant", name, "--seed", s, "--out", d,
                   "--model-info", info], R.dir("logs", "train"))


def emb_path(R, name, seed, ds, mode):
    return R.file("embed", f"{name}_s{seed}", f"{ds}_{mode}.pickle")


def stage_embed(R, datasets):
    info = R.file("model_info_full.json")
    if not os.path.isfile(info):
        mi, dc = cached_model_info(R)
        json.dump({"model_info": mi, "data_config": dc}, open(info, "w"))
    for name, s, ds, mode in all_runs(datasets):
        v = A.VARIANTS[name]
        cmd = [PY, "-m", "ablation.run_ablation", "_embed", "--variant", name, "--seed", s, "--dataset", ds,
               "--mode", mode, "--out", emb_path(R, name, s, ds, mode), "--model-info", info]
        if "objective" in v:
            ck = ckpt_of(R, name, s)
            if not os.path.isfile(ck) and not R.dry and not R.done(f"emb_{name}_s{s}_{ds}_{mode}"):
                R.log_time("embed", f"emb_{name}_s{s}_{ds}_{mode}", 0, "skipped: not trained")
                continue
            cmd += ["--ckpt", ck]
        R.run("embed", f"emb_{name}_s{s}_{ds}_{mode}", cmd, R.dir("logs", "embed"))


def sel_path(R, name, s, ds, mode):
    return R.file("selection", f"{name}_s{s}", f"{ds}_{mode}_l{A.LAMBDA}.pickle")


def stage_select(R, datasets):
    from utils_.select_datasets import Data_selection
    for name, s, ds, mode in all_runs(datasets):
        ep = emb_path(R, name, s, ds, mode)
        if not os.path.isfile(ep):
            continue
        E = pickle.load(open(ep, "rb"))
        sp = sel_path(R, name, s, ds, mode)
        # the original selection code, called on a stand-in for `self`
        ns = types.SimpleNamespace(vectors=E["lake"])
        dist = Data_selection.compute_distance(ns, vect_dict=E["lake"], vectors=E["queries"])
        sel = Data_selection.select_most_relevants_data(ns, dist, E["queries"], ratio=A.LAMBDA, compare="desc")
        pickle.dump(sel, open(sp, "wb"))
        RA.export_names(sp, sp.replace(".pickle", "_names.csv"))


def stage_opmodel(R, datasets):
    for name, s, ds, mode in all_runs(datasets):
        sp = sel_path(R, name, s, ds, mode)
        if not os.path.isfile(sp) and not R.dry:
            continue
        for op in A.DATASETS[ds]["operators"]:
            tag = f"{name}_s{s}_{ds}_{op}_{mode}"
            out = R.dir("opmodel", tag)
            json.dump(dict(variant=name, train_seed=s, dataset=ds, operator=op, mode=mode),
                      open(os.path.join(out, "meta.json"), "w"))
            R.run("opmodel", f"op_{tag}",
                  [PY, "operator_.py", "--dict-input", sp, "--out-path", out, "--out-file", "results.csv",
                   "--operator", op, "--data-type", "tabular", "--seeds", *A.SEEDS], R.dir("logs", "opmodel"))


def stage_table(R, datasets):
    import glob
    import pandas as pd
    runs = []
    for f in glob.glob(os.path.join(R.out, "opmodel", "*", "results.csv")):
        meta = os.path.join(os.path.dirname(f), "meta.json")
        if not os.path.isfile(meta):
            continue
        m = json.load(open(meta))
        if (m["variant"] not in A.VARIANTS or m["dataset"] not in datasets
                or m["operator"] not in A.DATASETS[m["dataset"]]["operators"]):
            continue
        df = pd.read_csv(f, on_bad_lines="skip")
        if "Seed" not in df.columns or df.empty:
            continue
        runs.append(df.assign(**m))
    if not runs:
        print("no operator results yet")
        return
    df = pd.concat(runs, ignore_index=True)
    metrics = ["RMSE", "MAE"]
    for c in metrics:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    keys = ["variant", "dataset", "operator", "mode"]
    per_seed = df.groupby(keys + ["train_seed", "Seed"])[metrics].mean().reset_index()
    agg = per_seed.groupby(keys)[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg["n_operator_seeds"] = per_seed.groupby(keys).size()
    agg = agg.reset_index()
    agg.insert(0, "dim", A.DIM)
    order = {v: i for i, v in enumerate(A.VARIANTS)}
    agg = agg.sort_values(["dataset", "operator", "mode", "variant"],
                          key=lambda c: c.map(order) if c.name == "variant" else c)
    agg.to_csv(R.file("ablation_results.csv"), index=False)

    parts = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(HERE, "out_d*", "ablation_results.csv")))]
    if parts:
        pd.concat(parts, ignore_index=True).to_csv(os.path.join(HERE, "ablation_results_all_dims.csv"), index=False)
    det = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(HERE, "out_d*", "dataset_detection.csv")))]
    if det:
        pd.concat(det, ignore_index=True).to_csv(os.path.join(HERE, "dataset_detection_all_dims.csv"), index=False)


def dataset_detection(R, name, datasets):
    """Can the encoder tell the datasets apart? Lake vectors of all datasets together;
    leave-one-out 1-NN (Euclidean): is the nearest other vector from the same dataset?
    Written before the embeddings are deleted -> dataset_detection.csv (one row per encoder)."""
    import pandas as pd
    rows = []
    for s in variant_seeds(name):
        for mode in modes_for(name):
            X, y = [], []
            for ds in datasets:
                ep = emb_path(R, name, s, ds, mode)
                if not os.path.isfile(ep):
                    continue
                L = pickle.load(open(ep, "rb"))["lake"]
                X += [np.asarray(v, float).ravel() for v in L.values()]
                y += [ds] * len(L)
            if len(set(y)) < 2:
                continue
            X, y = np.array(X), np.array(y)
            from scipy.spatial.distance import cdist
            D = cdist(X, X)
            np.fill_diagonal(D, np.inf)
            # ties (identical vectors) are broken at random, not by file order
            rng = np.random.default_rng(0)
            nn = np.array([rng.choice(np.flatnonzero(d == d.min())) for d in D])
            sizes = pd.Series(y).value_counts()
            rows.append(dict(dim=A.DIM, variant=name, train_seed=s, mode=mode, datasets=" ".join(sizes.index),
                             n_vectors=len(y), n_distinct_vectors=len(np.unique(X.round(12), axis=0)),
                             knn1_dataset_accuracy=float((y[nn] == y).mean()),
                             chance_majority=float(sizes.max() / len(y)),
                             mean_dist_within=float(np.mean([D[i][(y == y[i]) & np.isfinite(D[i])].mean()
                                                             for i in range(len(y)) if (y == y[i]).sum() > 1])),
                             mean_dist_between=float(np.mean([D[i][y != y[i]].mean() for i in range(len(y))]))))
    if not rows:
        return
    p = R.file("dataset_detection.csv")
    old = pd.read_csv(p) if os.path.isfile(p) else pd.DataFrame()
    if not old.empty:
        old = old[old.variant != name]
    pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(p, index=False)
    for r in rows:
        print(f"[detection] {name}: 1-NN dataset accuracy {r['knn1_dataset_accuracy']:.3f} "
              f"(majority {r['chance_majority']:.3f}), distinct vectors {r['n_distinct_vectors']}/{r['n_vectors']}",
              flush=True)


def op_tags(name, s, datasets):
    return [f"op_{name}_s{s}_{ds}_{op}_{mode}" for ds in datasets for mode in modes_for(name)
            for op in A.DATASETS[ds]["operators"]]


def free_disk(R, name, datasets):
    """Delete the encoder's weights (train/<name>_s<seed>/*.pt) and its embedding pickles, but only
    once every operator run of that encoder has finished and saved its results.csv. The
    selection pickles, the *_names.csv files, train_log.json and all results are kept."""
    import shutil
    for s in variant_seeds(name):
        if not all(R.done(t) for t in op_tags(name, s, datasets)):
            print(f"[cleanup] {name}_s{s}: weights kept (some operator runs did not finish)", flush=True)
            continue
        freed = 0
        tdir = os.path.join(R.out, "train", f"{name}_s{s}")
        for f in ("best.pt", "last.pt"):
            p = os.path.join(tdir, f)
            if os.path.isfile(p):
                freed += os.path.getsize(p)
                os.remove(p)
        edir = os.path.join(R.out, "embed", f"{name}_s{s}")
        if os.path.isdir(edir):
            freed += sum(os.path.getsize(os.path.join(edir, f)) for f in os.listdir(edir))
            shutil.rmtree(edir)
        if freed:
            print(f"[cleanup] {name}_s{s}: results saved, deleted weights + embeddings "
                  f"({freed / 2**30:.2f} GB)", flush=True)
            R.log_time("cleanup", f"cleanup_{name}_s{s}", 0, "ok", f"{freed} bytes")


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("_train", "_embed"):
        return worker(sys.argv[1:])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, f"out_d{A.DIM}"))
    ap.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    ap.add_argument("--datasets", nargs="+", default=list(A.DATASETS), choices=list(A.DATASETS))
    ap.add_argument("--variants", nargs="+", default=None, choices=list(A.VARIANTS),
                    help="subset of variants (default: all)")
    ap.add_argument("--keep-weights", action="store_true",
                    help="keep every trained encoder (default: delete its weights once its results are saved)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    R = RA.Runner(os.path.abspath(a.out), dry_run=a.dry_run)
    if a.variants:
        for k in [k for k in A.VARIANTS if k not in a.variants]:
            del A.VARIANTS[k]
    os.environ["VE_PATH_REMAP"] = ";".join(f"{o}=>{n}" for o, n in A.PATH_REMAP.items())
    order = [d for d in A.DATASETS if d in a.datasets]
    funcs = {"train": stage_train, "embed": stage_embed, "select": stage_select, "opmodel": stage_opmodel}
    stages = [s for s in STAGES if s in a.stages]
    t_all = time.time()
    if "setup" in stages:
        print("\n=== stage setup ===", flush=True)
        if not stage_setup(R, order):
            sys.exit("Setup check failed (see env.json).")
    # One encoder at a time: train -> embed -> select -> operators -> delete its weights,
    # so at most one encoder's weights are on disk at any moment.
    per_encoder = [s for s in stages if s in funcs]
    all_variants = dict(A.VARIANTS)
    for name, spec in all_variants.items():
        if not per_encoder:
            break
        A.VARIANTS = {name: spec}
        t0 = time.time()
        print(f"\n=== encoder {name} ===", flush=True)
        for st in per_encoder:
            print(f"--- {st}", flush=True)
            funcs[st](R, order)
        if not a.dry_run:
            dataset_detection(R, name, order)
        if not a.keep_weights and not a.dry_run:
            free_disk(R, name, order)
        R.log_time("encoder", name, time.time() - t0, "done")
    A.VARIANTS = all_variants
    if "table" in stages:
        print("\n=== stage table ===", flush=True)
        stage_table(R, order)
    R.log_time("all", "total", time.time() - t_all, "done")
    print(f"\nFinished. Table: {R.file('ablation_results.csv')}")


if __name__ == "__main__":
    main()
