import argparse
import hashlib
import json
import os
import re
import time

import numpy as np
import pandas as pd
import torch
from qdrant_client import QdrantClient, models

from utils_.selection_k import selection_k
from utils_.vectors import Vectorise

RESULTS_ROOT = os.environ.get("VENOM_RESULTS_ROOT", "results/")
DATA = os.path.join(RESULTS_ROOT, "data")
DATASETS = {
    #           evaluation folder               n_files  n_queries  |D|   k
    "hpc":     (f"{DATA}/HPC_2Krows/test",       401,     20,       381,  77),
    "adult":   (f"{DATA}/Adult/test",             99,     15,        84,  17),
    "stock":   (f"{DATA}/Stock/test_",           508,     10,       498, 100),
    "weather": (f"{DATA}/Weather/test",           49,     15,        34,   7),
}
CHECKPOINTS = {
    100: os.path.join(RESULTS_ROOT, "NumTabData2Vec_Weights", "Opt_adam_all_data_e100_v100", "best.pt"),
    200: os.path.join(RESULTS_ROOT, "NumTabData2Vec_Weights", "Opt_adam_all_data_e100_v200", "best.pt"),
    300: os.path.join(RESULTS_ROOT, "NumTabData2Vec_Weights", "Opt_adam_all_data_e100_v300", "best.pt"),
}
LAMBDA = 0.2
OUT = "results/nns_benchmark_paper_split"
PREFIX = "nnsbench_"          # only collections with this prefix are created / deleted
TIE_RTOL = 1e-5               # Qdrant computes in float32


# ----------------------------------------------------------------------------- data split
def chunk_number(path):
    nums = re.findall(r"\d+", os.path.basename(path))
    if not nums:
        raise ValueError(f"no chunk number in {path}")
    return int(nums[-1])


def file_hash(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def paper_split(ds):
    folder, n_files, n_queries, n_repo, k_expected = DATASETS[ds]
    files = sorted((os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".csv")),
                   key=chunk_number)
    nums = [chunk_number(f) for f in files]
    assert len(set(nums)) == len(nums), f"{ds}: duplicate chunk numbers in {folder}"
    assert len(files) == n_files, f"{ds}: {len(files)} evaluation files in {folder}, expected {n_files}"
    queries, repo = files[:n_queries], files[n_queries:]
    k = selection_k(LAMBDA, len(repo))

    assert len(queries) == n_queries, f"{ds}: {len(queries)} queries, expected {n_queries}"
    assert len(repo) == n_repo, f"{ds}: |D| = {len(repo)}, expected {n_repo}"
    assert k == k_expected, f"{ds}: k = {k}, expected {k_expected}"
    q_ids = {os.path.basename(f) for f in queries}
    d_ids = {os.path.basename(f) for f in repo}
    assert q_ids.isdisjoint(d_ids), f"{ds}: queries found in D: {sorted(q_ids & d_ids)}"

    q_hash = {file_hash(q): q for q in queries}
    same_content = [(os.path.basename(q_hash[h]), os.path.basename(f))
                    for f in repo if (h := file_hash(f)) in q_hash]

    def ids(lst):
        return f"{os.path.basename(lst[0])} .. {os.path.basename(lst[-1])}"
    print(f"{ds:8s} |D|={len(repo):4d}  k={k:4d}  #queries={len(queries):3d}  "
          f"queries: {ids(queries)}  D: {ids(repo)}")
    if same_content:
        print(f"  [warning] {ds}: {len(same_content)} file(s) in D have the same content as a query: "
              f"{same_content[:5]}")
    return queries, repo, k, same_content


def embed(files, ckpt, dim, eval_mode, seed, tmp_dir):
    os.makedirs(tmp_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    vec = Vectorise(model_path=ckpt, data_path=list(files), out_path=tmp_dir, data_type=1,
                    save_to="local", collection_name="unused", d_token=dim)
    if eval_mode:
        vec.model.eval()
    out = vec.compute_vectors(batch_size=1, show_progress=False, ret_vec=True)
    return np.stack([np.asarray(out[f], dtype=np.float64).ravel() for f in files])


def load_or_embed(ds, dim, mode, queries, lake, args):
    vdir = os.path.join(args.out, "vectors")
    os.makedirs(vdir, exist_ok=True)
    stem = os.path.join(vdir, f"{ds}_d{dim}_{mode}")
    paths = {p: f"{stem}_{p}.npy" for p in ("lake", "queries", "lake_names", "queries_names")}
    if not args.reembed and all(os.path.isfile(p) for p in paths.values()):
        if (list(np.load(paths["lake_names"])) == lake
                and list(np.load(paths["queries_names"])) == queries):
            print(f"  reuse {stem}_*.npy")
            return np.load(paths["lake"]), np.load(paths["queries"])
        print("  saved names differ from the current split -> re-embedding")
    t0 = time.time()
    L = embed(lake, CHECKPOINTS[dim], dim, eval_mode=(mode == "eval"), seed=args.seed,
              tmp_dir=os.path.join(args.out, "logs"))
    Q = embed(queries, CHECKPOINTS[dim], dim, eval_mode=True, seed=args.seed,
              tmp_dir=os.path.join(args.out, "logs"))
    print(f"  embedded {len(lake)} D + {len(queries)} queries in {time.time() - t0:.1f}s")
    np.save(paths["lake"], L)
    np.save(paths["queries"], Q)
    np.save(paths["lake_names"], np.array(lake))
    np.save(paths["queries_names"], np.array(queries))
    return L, Q


def wait_indexed(client, name, need_index, timeout=600):
    t0 = time.time()
    while True:
        info = client.get_collection(name)
        idx = info.indexed_vectors_count or 0
        if str(info.status).lower().endswith("green") and (not need_index or idx >= info.points_count):
            return info
        if time.time() - t0 > timeout:
            print(f"  [!] {name}: index not complete after {timeout}s (indexed {idx}/{info.points_count})")
            return info
        time.sleep(0.5)


def build_collection(client, name, L, hnsw=None):
    if client.collection_exists(name):
        client.delete_collection(name)
    kw = {}
    if hnsw is not None:
        m, efc = hnsw
        kw = dict(hnsw_config=models.HnswConfigDiff(m=m, ef_construct=efc, full_scan_threshold=10),
                  optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1, default_segment_number=1))
    client.create_collection(name, vectors_config=models.VectorParams(size=L.shape[1],
                                                                      distance=models.Distance.EUCLID), **kw)
    client.upsert(name, wait=True, points=[models.PointStruct(id=i, vector=v.tolist())
                                           for i, v in enumerate(L.astype(np.float32))])
    return wait_indexed(client, name, need_index=hnsw is not None)


def index_params(info, name, hnsw):
    cfg = info.config
    return dict(collection=name, forced_hnsw=hnsw is not None,
                points_count=info.points_count, indexed_vectors_count=info.indexed_vectors_count,
                segments_count=info.segments_count, distance=str(cfg.params.vectors.distance),
                hnsw_m=cfg.hnsw_config.m, hnsw_ef_construct=cfg.hnsw_config.ef_construct,
                hnsw_full_scan_threshold_kb=cfg.hnsw_config.full_scan_threshold,
                indexing_threshold_kb=cfg.optimizer_config.indexing_threshold,
                default_segment_number=cfg.optimizer_config.default_segment_number,
                quantization=cfg.quantization_config is not None)


def qdrant_topk(client, name, Q, k, hnsw_ef=None, exact=False):
    params = models.SearchParams(hnsw_ef=hnsw_ef, exact=exact) if (hnsw_ef or exact) else None
    ids, times = [], []
    for q in Q.astype(np.float32):
        t0 = time.perf_counter()
        res = client.query_points(name, query=q.tolist(), limit=k, search_params=params).points
        times.append(time.perf_counter() - t0)
        ids.append([p.id for p in res])
    return ids, times


def exact_distances(Q, L):
    D = np.sqrt(((Q[:, None, :].astype(np.float64) - L[None, :, :].astype(np.float64)) ** 2).sum(-1))
    return D, np.argsort(D, axis=1, kind="stable")


def recall_per_query(D, order, approx_ids, k):
    rid, rtie = [], []
    for q, ids in enumerate(approx_ids):
        ex = set(order[q, :k].tolist())
        dk = D[q, order[q, k - 1]]
        ids = list(ids)[:k]
        rid.append(len(ex & set(ids)) / k)
        rtie.append(sum(D[q, i] <= dk * (1 + TIE_RTOL) + 1e-12 for i in ids) / k)
    return rid, rtie


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--dims", nargs="+", type=int, default=[100, 200, 300])
    ap.add_argument("--modes", nargs="+", default=["paper"], choices=["paper", "eval"])
    ap.add_argument("--m", nargs="+", type=int, default=[4, 8, 16])
    ap.add_argument("--ef-construct", nargs="+", type=int, default=[16, 100])
    ap.add_argument("--hnsw-ef", nargs="+", type=int, default=[8, 128])
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reembed", action="store_true")
    ap.add_argument("--keep-collections", action="store_true")
    ap.add_argument("--embed-only", action="store_true", help="only compute / save the .npy vectors")
    ap.add_argument("--check-split", action="store_true", help="only build, check and print the split")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    # every split is built and checked before anything is embedded or searched
    print("Split (queries = first n evaluation files by chunk number; D = the rest):")
    splits = {ds: paper_split(ds) for ds in args.datasets}
    if args.check_split:
        return
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "split.json"), "w") as f:
        json.dump({ds: dict(folder=DATASETS[ds][0], n_queries=len(q), n_repository=len(d), k=k,
                            same_content_as_query=sc, queries=q, repository=d)
                   for ds, (q, d, k, sc) in splits.items()}, f, indent=1)

    client = None if args.embed_only else QdrantClient(url=args.qdrant_url, timeout=300)
    per_query, params = [], []
    for ds in args.datasets:
        queries, repo, k, _ = splits[ds]
        print(f"\n=== {ds}: {len(queries)} queries, |D| = {len(repo)}, k = {k}")
        for dim in args.dims:
            for mode in args.modes:
                print(f" -- d={dim} mode={mode}")
                L, Q = load_or_embed(ds, dim, mode, queries, repo, args)
                if args.embed_only:
                    continue
                D, order = exact_distances(Q.astype(np.float32), L.astype(np.float32))
                t0 = time.perf_counter()
                exact_distances(Q.astype(np.float32), L.astype(np.float32))
                t_exact = (time.perf_counter() - t0) / len(Q)

                configs = [("pipeline_default", None)] + [(f"hnsw_m{m}_efc{e}", (m, e))
                                                          for m in args.m for e in args.ef_construct]
                for cname, hnsw in configs:
                    name = f"{PREFIX}{ds}_d{dim}_{mode}_{cname}"
                    info = build_collection(client, name, L, hnsw)
                    params.append(dict(dataset=ds, dim=dim, mode=mode, config=cname,
                                       **index_params(info, name, hnsw)))
                    searches = [("exact=True", None, True)]
                    searches += ([("default", None, False)] if hnsw is None
                                 else [(f"ef={ef}", ef, False) for ef in args.hnsw_ef])
                    for sname, ef, exact in searches:
                        ids, times = qdrant_topk(client, name, Q, k, hnsw_ef=ef, exact=exact)
                        rid, rtie = recall_per_query(D, order, ids, k)
                        for qi in range(len(Q)):
                            per_query.append(dict(
                                dataset=ds, dim=dim, mode=mode, n_repository=len(repo), k=k,
                                config=cname, search=sname, hnsw_ef=ef,
                                effective_ef=max(ef, k) if ef else None,
                                indexed_vectors=info.indexed_vectors_count, total_vectors=info.points_count,
                                query=os.path.basename(queries[qi]),
                                recall_id=rid[qi], recall_tie=rtie[qi],
                                qdrant_ms=1e3 * times[qi], exact_numpy_ms_per_query=1e3 * t_exact))
                        print(f"    {cname:18s} {sname:10s} recall={np.mean(rid):.3f} "
                              f"ties={np.mean(rtie):.3f} indexed={info.indexed_vectors_count}/{info.points_count}")
                    if not args.keep_collections:
                        client.delete_collection(name)
                save(per_query, params, args.out)

    if per_query:
        save(per_query, params, args.out, final=True)
    print(f"\nwritten to {args.out}")


def save(per_query, params, out, final=False):
    pq = pd.DataFrame(per_query)
    pq.to_csv(os.path.join(out, "per_query.csv"), index=False)
    pd.DataFrame(params).to_csv(os.path.join(out, "index_params.csv"), index=False)
    keys = ["dataset", "dim", "mode", "n_repository", "k", "config", "search", "hnsw_ef", "effective_ef",
            "indexed_vectors", "total_vectors"]
    agg = (pq.groupby(keys, dropna=False, sort=False)
             .agg(n_queries=("query", "size"), recall_id=("recall_id", "mean"),
                  recall_tie=("recall_tie", "mean"), qdrant_ms_per_query=("qdrant_ms", "mean"),
                  exact_numpy_ms_per_query=("exact_numpy_ms_per_query", "first"))
             .reset_index())
    agg.to_csv(os.path.join(out, "recall.csv"), index=False)
    if final:
        write_table(agg, out)


def write_table(agg, out):
    """One row per dataset x vector size x configuration (exact=True sanity rows left out)."""
    t = agg[agg.search != "exact=True"]
    lines = [r"\begin{tabular}{lrrlllrrr}", r"\toprule",
             r"Dataset & $d$ & $k$ & Index & Search & Indexed & Recall@$k$ & Recall@$k$ (ties) & ms/query \\",
             r"\midrule"]
    for _, r in t.iterrows():
        lines.append(f"{r.dataset} & {r.dim} & {r.k} & {r.config.replace('_', ' ')} & {r.search} & "
                     f"{r.indexed_vectors}/{r.total_vectors} & {r.recall_id:.3f} & {r.recall_tie:.3f} & "
                     f"{r.qdrant_ms_per_query:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(out, "nns_benchmark.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
