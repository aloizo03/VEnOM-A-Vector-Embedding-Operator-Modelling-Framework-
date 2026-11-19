import argparse
import csv
import glob
import json
import os
import pickle
import platform
import shutil
import subprocess
import sys
import tarfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
if os.environ.get("VE_RERUN_CONFIG"):
    import importlib.util
    _spec = importlib.util.spec_from_file_location("config", os.environ["VE_RERUN_CONFIG"])
    C = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(C)
else:
    import config as C  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, ".."))
PY = sys.executable
STAGES = ["setup", "collections", "verify", "grid", "lambda", "sr", "embedders"]



class Runner:
    def __init__(self, out, dry_run=False):
        self.out = out
        self.dry = dry_run
        os.makedirs(out, exist_ok=True)
        self.runtime_csv = os.path.join(out, "runtime.csv")

    def dir(self, *p):
        """<out>/p..., created as a directory."""
        full = os.path.join(self.out, *p)
        os.makedirs(full, exist_ok=True)
        return full

    def file(self, *p):
        """<out>/p..., with its parent directory created."""
        full = os.path.join(self.out, *p)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        return full

    def done(self, tag):
        return os.path.isfile(os.path.join(self.out, "_done", tag))

    def mark(self, tag):
        os.makedirs(os.path.join(self.out, "_done"), exist_ok=True)
        open(os.path.join(self.out, "_done", tag), "w").write(time.strftime("%F %T"))

    def log_time(self, stage, tag, seconds, status, extra=""):
        new = not os.path.isfile(self.runtime_csv)
        with open(self.runtime_csv, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["time", "stage", "step", "wall_time_s", "status", "extra"])
            w.writerow([time.strftime("%F %T"), stage, tag, f"{seconds:.3f}", status, extra])

    def run(self, stage, tag, cmd, log_dir):
        if self.done(tag):
            return True
        os.makedirs(log_dir, exist_ok=True)
        print(f"[{stage}] {tag}", flush=True)
        if self.dry:
            print("    " + " ".join(map(str, cmd)))
            return True
        t0 = time.time()
        with open(os.path.join(log_dir, f"{tag}.log"), "w") as fh:
            rc = subprocess.run(list(map(str, cmd)), cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode
        dt = time.time() - t0
        self.log_time(stage, tag, dt, "ok" if rc == 0 else f"exit {rc}")
        if rc == 0:
            self.mark(tag)
        else:
            print(f"    FAILED (exit {rc}) after {dt:.0f}s -- see {log_dir}/{tag}.log", flush=True)
        return rc == 0


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable: {e}"


def qclient():
    from qdrant_client import QdrantClient
    return QdrantClient(url=C.QDRANT_URL, timeout=300)


def newest(pattern):
    m = glob.glob(pattern)
    return max(m, key=os.path.getmtime) if m else None



def stage_setup(R, datasets):
    env = {"time": time.strftime("%F %T"), "host": platform.node(), "platform": platform.platform(),
           "python": sys.version, "cpu": sh("lscpu | grep -E 'Model name|^CPU\\(s\\)'"),
           "memory": sh("free -g | head -2"), "gpu": sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv"),
           "git_commit": sh(f"git -c safe.directory={REPO} -C {REPO} rev-parse HEAD"), "git_branch": sh(f"git -c safe.directory={REPO} -C {REPO} branch --show-current"),
           "git_dirty": sh(f"git -c safe.directory={REPO} -C {REPO} status --porcelain --untracked-files=no"), "packages": {}}
    for mod in ["numpy", "pandas", "sklearn", "scipy", "torch", "qdrant_client", "networkx", "transformers",
                "permetrics", "statsmodels"]:
        try:
            m = __import__(mod)
            env["packages"][mod] = getattr(m, "__version__", "?")
        except Exception as e:  # noqa: BLE001
            env["packages"][mod] = f"MISSING ({e})"
    env["torch_cuda"] = sh(f"{PY} -c \"import torch;print(torch.cuda.is_available(), torch.version.cuda)\"")
    checks = []
    try:
        c = qclient()
        env["qdrant_version"] = sh(f"curl -s {C.QDRANT_URL}")
        names = {x.name for x in c.get_collections().collections}
        checks.append(("qdrant", C.QDRANT_URL, "OK"))
    except Exception as e:  # noqa: BLE001
        names = set()
        checks.append(("qdrant", C.QDRANT_URL, f"UNREACHABLE: {e}"))
    for ds in datasets:
        cfg = C.DATASETS[ds]
        for key in ("lake", "queries"):
            checks.append((ds, cfg[key], "OK" if os.path.exists(cfg[key]) else "MISSING"))
        for dim, coll in cfg["collections"].items():
            checks.append((ds, f"collection {dim}: {coll}", "OK" if coll in names else "MISSING (will rebuild)"))
        for emb, coll in cfg.get("embedders", {}).items():
            checks.append((ds, f"collection {emb}: {coll}", "OK" if coll in names else "MISSING (will rebuild)"))
    for p in list(C.NUM2VEC_CKPT.values()) + [C.GRAPH_CKPT]:
        checks.append(("checkpoint", p, "OK" if os.path.exists(p) else "MISSING"))
    json.dump(env, open(R.file("env.json"), "w"), indent=1)
    with open(R.file("setup_check.csv"), "w", newline="") as fh:
        csv.writer(fh).writerows([("scope", "item", "status")] + checks)
    bad = [c for c in checks if c[2].startswith(("MISSING", "UNREACHABLE")) and "will rebuild" not in c[2]]
    for c in checks:
        print(f"  {c[2]:<24} {c[0]:<10} {c[1]}")
    print(f"GPU: {env['gpu']}\nCPU: {env['cpu']}\ngit: {env['git_branch']} {env['git_commit'][:10]}"
          + (" (DIRTY)" if env["git_dirty"] else ""))
    if env["git_branch"] != "revision-fixes":
        print("WARNING: not on branch revision-fixes -- the fixes would not be applied.")
    return not bad



def collection_points(name):
    try:
        return qclient().get_collection(name).points_count
    except Exception:  # noqa: BLE001
        return None


def collections_map(R):
    f = R.file("collections_used.csv")
    if not os.path.isfile(f):
        return {}
    return {(r["dataset"], r["key"]): r for r in csv.DictReader(open(f))}


def record_collection(R, row):
    f = R.file("collections_used.csv")
    rows = collections_map(R)
    rows[(row["dataset"], row["key"])] = row
    with open(f, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "key", "requested", "used", "rebuilt", "points",
                                           "vectorise_wall_time_s"])
        w.writeheader()
        w.writerows(rows.values())


def rebuild_collection(R, ds, key, dim, tab_model=None):
    cfg = C.DATASETS[ds]
    name = f"rerun_{ds}_{key}"
    cmd = [PY, "datasets_to_vec.py", "-i", cfg["lake"], "-out", R.dir("vectorisation", f"{ds}_{key}"),
           "-cn", name, "-dt", cfg["type"], "-s", "vectorDB", "-bz", "16"]
    if tab_model:
        cmd += ["-tm", tab_model]
    else:
        cmd += ["-vs", str(dim)]
        if cfg["type"] == "tabular":
            cmd += ["-ckpt", C.NUM2VEC_CKPT[dim]]
        elif cfg["type"] == "graph":
            cmd += ["-ckpt", C.GRAPH_CKPT]
    t0 = time.time()
    ok = R.run("collections", f"vectorise_{ds}_{key}", cmd, R.dir("logs", "collections"))
    if not ok or R.dry:
        return None, None
    made = sorted(x.name for x in qclient().get_collections().collections if x.name.startswith(f"vectors_{name}_"))
    if not made:
        return None, time.time() - t0
    # datasets_to_vec.py upserts with wait=False: wait until the point count is stable
    last, stable = -1, 0
    while stable < 3:
        time.sleep(2)
        n = collection_points(made[-1]) or 0
        stable = stable + 1 if n == last else 0
        last = n
    return made[-1], time.time() - t0


def stage_collections(R, datasets, measure=False):
    """Use the original collection when it exists; otherwise rebuild it from the raw lake."""
    used = collections_map(R)
    for ds in datasets:
        cfg = C.DATASETS[ds]
        wanted = [(str(d), d, c, None) for d, c in cfg["collections"].items()]
        wanted += [(emb, None, c, emb) for emb, c in cfg.get("embedders", {}).items()]
        for key, dim, coll, emb in wanted:
            if (ds, key) in used and used[(ds, key)]["used"]:
                continue
            pts = collection_points(coll) if coll else None
            if pts:
                record_collection(R, {"dataset": ds, "key": key, "requested": coll, "used": coll, "rebuilt": "no",
                                      "points": pts, "vectorise_wall_time_s": ""})
                if not measure:
                    continue
            new, secs = rebuild_collection(R, ds, key, dim, tab_model=emb)
            if pts:  # measurement-only rebuild of an existing collection: keep using the original
                record_collection(R, {"dataset": ds, "key": key, "requested": coll, "used": coll, "rebuilt": "no",
                                      "points": pts, "vectorise_wall_time_s": f"{secs:.3f}" if secs else ""})
            else:
                record_collection(R, {"dataset": ds, "key": key, "requested": coll or "(none logged)",
                                      "used": new or "", "rebuilt": "yes" if new else "FAILED",
                                      "points": collection_points(new) if new else "",
                                      "vectorise_wall_time_s": f"{secs:.3f}" if secs else ""})


def coll_for(R, ds, key):
    row = collections_map(R).get((ds, str(key)))
    return row["used"] if row and row["used"] else None



def query_cache(R, ds, key):
    return R.dir("query_cache", f"{ds}_{key}")


def select(R, stage, ds, key, dim, method, lam, tab_model=None):
    """select_data.py -> returns the selection pickle (and exports names to CSV)."""
    cfg = C.DATASETS[ds]
    coll = coll_for(R, ds, key)
    tag = f"sel_{ds}_{key}_{method}_l{lam}"
    out = R.dir("selection", tag)
    if coll is None:
        R.log_time(stage, tag, 0, "skipped: no collection")
        return None
    cmd = [PY, "select_data.py", "-i", cfg["queries"], "-out", out, "-v", coll, "-s", C.METHODS[method],
           "-r", lam, "-dt", cfg["type"], "-vb", "--query-cache", query_cache(R, ds, key)]
    if tab_model:
        cmd += ["-tm", tab_model, "-vs", "32" if tab_model == "dataset2vec" else "300"]
    else:
        cmd += ["-vs", dim]
        if cfg["type"] == "tabular":
            cmd += ["-ckpt", C.NUM2VEC_CKPT[dim]]
        elif cfg["type"] == "graph":
            cmd += ["-ckpt", C.GRAPH_CKPT]
    if not R.run(stage, tag, cmd, R.dir("logs", stage)):
        return None
    pk = newest(os.path.join(out, f"rel_data_{C.METHODS[method]}_*.pickle"))
    if pk and not os.path.isfile(os.path.join(out, "selection_names.csv")):
        export_names(pk, os.path.join(out, "selection_names.csv"))
    return pk


def export_names(pickle_path, csv_path):
    d = pickle.load(open(pickle_path, "rb"))
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["query", "rank", "dataset"])
        for q, (names, _) in d["selected_dataset"].items():
            for i, n in enumerate(names):
                w.writerow([q, i, n])


def labels(R, ds, op):
    """(lake labels, query labels) for graph/image; (None, None) for tabular."""
    cfg = C.DATASETS[ds]
    if cfg["type"] == "tabular":
        return None, None
    out = {}
    for side, src in (("lake", cfg["lake"]), ("query", cfg["queries"])):
        if side == "query" and cfg["type"] == "image":
            out[side] = cfg["queries"]  # the query CSV already holds the class labels
            continue
        old = os.path.join(C.OLD_LABELS_DIR, f"{ds}_{op}_{side}", "scores.csv")
        if os.path.isfile(old):
            out[side] = old
            continue
        d = R.dir("labels", f"{ds}_{op}_{side}")
        cmd = [PY, "train_exp.py", "-i", src, "-out", d, "-o", op, "-q", "all", "-dt", cfg["type"], "-fn", "scores"]
        R.run("labels", f"labels_{ds}_{op}_{side}", cmd, R.dir("logs", "labels"))
        out[side] = newest(os.path.join(d, "scores*"))
    return out["lake"], out["query"]


def opmodel(R, stage, ds, op, sel_pickle, tag):
    if sel_pickle is None:
        R.log_time(stage, tag, 0, "skipped: no selection")
        return None
    cfg = C.DATASETS[ds]
    out = R.dir("opmodel", tag)
    cmd = [PY, "operator_.py", "--dict-input", sel_pickle, "--out-path", out, "--out-file", "results.csv",
           "--operator", op, "--data-type", cfg["type"], "--seeds", *C.SEEDS]
    lake_lab, q_lab = labels(R, ds, op)
    if lake_lab:
        cmd += ["--operator-input", lake_lab]
    if q_lab:
        cmd += ["--target-labels", q_lab]
    R.run(stage, f"op_{tag}", cmd, R.dir("logs", stage))
    return os.path.join(out, "results.csv")



def stage_verify(R, datasets):
    import pandas as pd
    from qdrant_client import models
    sys.path.insert(0, REPO)
    from utils_.selection_k import selection_k
    from utils_.path_remap import remap_path

    rows, ok_all = [], True
    for ds, op in C.VERIFY:
        if ds not in datasets:
            continue
        dim = C.DATASETS[ds]["main_dim"]
        coll = coll_for(R, ds, dim)
        pk_dist = select(R, "verify", ds, dim, dim, "distance", C.LAMBDA)
        pk_vdb = select(R, "verify", ds, dim, dim, "vectordb", C.LAMBDA)
        if R.dry:
            continue
        if not (pk_dist and pk_vdb and coll):
            # fail closed: a verification that could not run is not a pass
            ok_all = False
            rows.append({"dataset": ds, "operator": op, "dim": dim, "collection": coll, "PASS": False,
                         "note": "could not run: " + ", ".join(n for n, x in (("collection", coll), (
                             "distance selection", pk_dist), ("vectordb selection", pk_vdb)) if not x)
                         + " missing (see logs/verify)"})
            print(f"  verify {ds}/{op}: COULD NOT RUN -> FAIL")
            continue
        dist = pickle.load(open(pk_dist, "rb"))
        vdb = pickle.load(open(pk_vdb, "rb"))
        c = qclient()
        n = c.get_collection(coll).points_count
        k = selection_k(C.LAMBDA, n, rounding='floor')
        exact = {"selected_dataset": {}, "Pred_Dataset": dist["Pred_Dataset"]}
        same_names = same_vecs = same_vdb = 0
        for q, qv in dist["Pred_Dataset"].items():
            pts = c.query_points(coll, query=np.asarray(qv, dtype=float).tolist(), limit=k, with_vectors=True,
                                 with_payload=True, search_params=models.SearchParams(exact=True)).points
            en = [remap_path(p.payload["dataset_name"]) for p in pts]
            ev = [p.vector for p in pts]
            exact["selected_dataset"][q] = [en, ev]
            hn, hv = dist["selected_dataset"][q]
            same_names += en == list(hn)
            same_vecs += len(ev) == len(hv) and all(np.allclose(a, b) for a, b in zip(ev, hv))
            same_vdb += list(vdb["selected_dataset"][q][0]) == list(hn)
        nq = len(dist["Pred_Dataset"])
        pk_exact = R.file("selection", f"verify_exact_{ds}_{dim}", "rel_data_exact.pickle")
        pickle.dump(exact, open(pk_exact, "wb"))
        export_names(pk_exact, os.path.join(os.path.dirname(pk_exact), "selection_names.csv"))
        res = {}
        for name, pk in (("handwritten", pk_dist), ("exact", pk_exact)):
            f = opmodel(R, "verify", ds, op, pk, f"verify_{ds}_{op}_{dim}_{name}")
            res[name] = pd.read_csv(f) if f and os.path.isfile(f) else None
        same_err, per_seed = False, ""
        if res["handwritten"] is not None and res["exact"] is not None:
            a, b = res["handwritten"], res["exact"]
            same_err = len(a) == len(b) and np.allclose(pd.to_numeric(a.RMSE), pd.to_numeric(b.RMSE), equal_nan=True)
            per_seed = ";".join(
                f"{s}:{a[a.Seed == s].RMSE.astype(float).mean():.6g}/{b[b.Seed == s].RMSE.astype(float).mean():.6g}"
                for s in sorted(a.Seed.unique()))
        ok = same_names == nq and same_vecs == nq and bool(same_err)
        ok_all &= ok
        rows.append({"dataset": ds, "operator": op, "dim": dim, "collection": coll, "N": n, "k": k,
                     "queries": nq, "same_ranked_names_exact_vs_handwritten": same_names,
                     "same_vectors_exact_vs_handwritten": same_vecs,
                     "same_ranked_names_default_vectordb_vs_handwritten": same_vdb,
                     "identical_errors_all_seeds": same_err, "per_seed_RMSE_handwritten/exact": per_seed,
                     "PASS": ok})
        print(f"  verify {ds}/{op}: names {same_names}/{nq}, vectors {same_vecs}/{nq}, "
              f"errors identical={same_err} -> {'PASS' if ok else 'FAIL'}")
    if rows:
        pd.DataFrame(rows).to_csv(R.file("verification.csv"), index=False)
    return ok_all



def stage_grid(R, datasets):
    for ds in datasets:
        cfg = C.DATASETS[ds]
        for dim in cfg["collections"]:
            for method in C.METHODS:
                pk = select(R, "grid", ds, dim, dim, method, C.LAMBDA)
                for op in cfg["operators"]:
                    opmodel(R, "grid", ds, op, pk, f"grid_{ds}_{op}_{dim}_{method}")


def stage_lambda(R, datasets):
    for ds in datasets:
        cfg = C.DATASETS[ds]
        dim = cfg["main_dim"]
        for lam in C.LAMBDA_SWEEP:
            for method in C.METHODS:
                pk = select(R, "lambda", ds, dim, dim, method, lam)
                for op in cfg["operators"]:
                    opmodel(R, "lambda", ds, op, pk, f"lambda_{ds}_{op}_{dim}_{method}_l{lam}")


def stage_sr(R, datasets):
    for ds in datasets:
        cfg = C.DATASETS[ds]
        if not cfg.get("sr"):
            continue
        dim = cfg["main_dim"]
        coll = coll_for(R, ds, dim)
        if coll is None:
            continue
        for op in cfg["operators"]:
            for sr in C.SR_RATIOS:
                tag = f"sr_{ds}_{op}_{dim}_sr{sr}"
                cmd = [PY, "train_op_sr.py", "-v", coll, "-vb", "-p", cfg["queries"], "-out", R.dir("sr", tag),
                       "-ckpt", C.NUM2VEC_CKPT[dim], "-m", C.SR_MODEL, "-sr", sr, "-q", "last", "-o", op,
                       "--seeds", *C.SEEDS]
                R.run("sr", tag, cmd, R.dir("logs", "sr"))


def stage_embedders(R, datasets):
    for ds in datasets:
        cfg = C.DATASETS[ds]
        for emb in cfg.get("embedders", {}):
            for method in C.METHODS:
                pk = select(R, "embedders", ds, emb, None, method, C.LAMBDA, tab_model=emb)
                for op in cfg["operators"]:
                    opmodel(R, "embedders", ds, op, pk, f"emb_{ds}_{op}_{emb}_{method}")



def pack(R):
    """tar.gz of everything needed for rerun/make_report.py (no pickles / query caches)."""
    dest = R.out.rstrip("/") + f"_{platform.node()}_{time.strftime('%Y%m%d_%H%M')}.tar.gz"
    skip_dirs = {"query_cache", "_done"}

    def keep(ti):
        parts = ti.name.split("/")
        if any(p in skip_dirs for p in parts) or ti.name.endswith((".pickle", ".pkl", ".pt", ".tmp")):
            return None
        return ti
    with tarfile.open(dest, "w:gz") as tf:
        tf.add(R.out, arcname=os.path.basename(R.out.rstrip("/")), filter=keep)
    print(f"Packed -> {dest} ({os.path.getsize(dest) / 1e6:.1f} MB). Send this file back.")



def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    ap.add_argument("--datasets", nargs="+", default=list(C.DATASETS), choices=list(C.DATASETS))
    ap.add_argument("--measure-vectorization", action="store_true",
                    help="also re-vectorise EXISTING collections into throw-away ones, only to time T_vec_repo")
    ap.add_argument("--continue-on-verify-fail", action="store_true")
    ap.add_argument("--ignore-setup-errors", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    ap.add_argument("--pack", action="store_true", help="only pack <out> into a tar.gz and exit")
    a = ap.parse_args()
    R = Runner(os.path.abspath(a.out), dry_run=a.dry_run)
    if a.pack:
        return pack(R)
    order = [d for d in C.DATASETS if d in a.datasets]
    os.environ["VE_QUERY_CACHE"] = ""  # caching is passed explicitly per (dataset, embedder, dim)
    # dataset names stored in the March collections point at a data root that has moved;
    # inherited by every subprocess (utils_/path_remap.py)
    os.environ["VE_PATH_REMAP"] = ";".join(f"{o}=>{n}" for o, n in getattr(C, "PATH_REMAP", {}).items())
    t_all = time.time()
    for st in [s for s in STAGES if s in a.stages]:
        t0 = time.time()
        print(f"\n=== stage {st} ===", flush=True)
        if st == "setup":
            if not stage_setup(R, order) and not a.ignore_setup_errors:
                sys.exit("Setup check failed (see setup_check.csv). Fix it or pass --ignore-setup-errors.")
        elif st == "collections":
            stage_collections(R, order, measure=a.measure_vectorization)
        elif st == "verify":
            if not stage_verify(R, order) and not R.dry and not a.continue_on_verify_fail:
                R.log_time(st, "stage", time.time() - t0, "FAILED")
                sys.exit("VERIFICATION FAILED: Qdrant exact and hand-written Euclidean differ "
                         "(see verification.csv). Stopping as requested.")
        else:
            {"grid": stage_grid, "lambda": stage_lambda, "sr": stage_sr, "embedders": stage_embedders}[st](R, order)
        R.log_time(st, "stage", time.time() - t0, "done")
    R.log_time("all", "total", time.time() - t_all, "done")
    print(f"\nFinished. Results in {R.out}. Run `python rerun/run_all.py --pack` and send the tar.gz back.")


if __name__ == "__main__":
    main()
