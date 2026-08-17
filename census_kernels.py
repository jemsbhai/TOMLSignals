"""
TOMLSignals - EXP-CR-005: Kernel-Launch Census
===============================================
Counts CUDA kernel launches (and memcpy / memset) per algorithm invocation
with torch.profiler for every (algorithm, N, B) configuration present in the
RTX 4090 and A100 result sets, and compares the counts with the model's
launch counts (shared.to_model.get_seq_steps / KERNELS_PER_ITER).

Why: the model's dispatch term alpha_o * S_o uses hand-derived launch counts
(F-013). F-020 found KERNELS_PER_ITER["iir_butter4"] = 5 where the fallback
code issues ~22 launches per sample, and found that cuSOLVER-internal launches
(svd, pca) cost the same per launch as Python-loop launches but are not
counted. This census measures every launch count the model uses.

No energy is measured. Nothing in shared/, algorithms/ or the v0 analysis is
modified. Resumable: rows are appended to the CSV as they complete; already
completed configurations are skipped unless --force.

Backends (--backend auto|torch|nsys):
  torch  torch.profiler CUDA activity (needs Kineto built with CUPTI; the
         Windows torch 2.6 wheel is CPU-only, so this fails there)
  nsys   Nsight Systems: each configuration is run in a child process under
         `nsys profile --trace=cuda --capture-range=cudaProfilerApi`; the child
         (this same script with --worker) sets up, warms up, and brackets ONE
         invocation with cudaProfilerStart/Stop; kernels are then counted from
         `nsys stats --report cuda_gpu_kern_sum` and memcpy/memset from
         cuda_gpu_mem_time_sum. Exact counts, negligible overhead.

Usage (repo root):
  python census_kernels.py                 # all configurations
  python census_kernels.py --algs svd pca  # subset
  python census_kernels.py --summary-only  # re-print tables from the CSV

Outputs: data/camera_ready/exp_cr_005_kernel_census.csv
         data/camera_ready/exp_cr_005_kernel_census.json

Author: Muntaser Syed
Date: August 2026
"""

import argparse
import csv
import glob
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import torch  # noqa: E402
from torch.profiler import profile, ProfilerActivity  # noqa: E402
from torch.autograd import DeviceType  # noqa: E402

from algorithms.transforms import TRANSFORMS  # noqa: E402
from algorithms.filters import FILTERS  # noqa: E402
from algorithms.adaptive import ADAPTIVE  # noqa: E402
from algorithms.estimation import ESTIMATION  # noqa: E402
from algorithms.spectral import SPECTRAL  # noqa: E402
from algorithms.decomposition import DECOMPOSITION  # noqa: E402
from algorithms.compression import COMPRESSION  # noqa: E402
from algorithms.ml_enhanced import ML_ENHANCED  # noqa: E402
from shared.to_model import get_seq_steps, get_fused_steps, SEQ_STEPS, KERNELS_PER_ITER  # noqa: E402
import analyze_results as ar  # noqa: E402  (v0 loaders, unmodified)

OUT_DIR = BASE / "data" / "camera_ready"
CSV_OUT = OUT_DIR / "exp_cr_005_kernel_census.csv"
JSON_OUT = OUT_DIR / "exp_cr_005_kernel_census.json"

CATEGORIES = {
    "transform": TRANSFORMS, "filter": FILTERS, "adaptive": ADAPTIVE,
    "estimation": ESTIMATION, "spectral": SPECTRAL, "decomposition": DECOMPOSITION,
    "compression": COMPRESSION, "ml_enhanced": ML_ENHANCED,
}
ALL_ALGORITHMS = {}
CATEGORY_OF = {}
for _cat, _d in CATEGORIES.items():
    for _name, _v in _d.items():
        ALL_ALGORITHMS[_name] = _v
        CATEGORY_OF[_name] = _cat

# (label, results csv, has_torchaudio on that GPU, IIR rerun dir)
GPU_SOURCES = [
    ("RTX 4090", "data/results/all_results.csv", True, None),
    ("A100 SXM4", "data/server_results/results/all_results.csv", False,
     "data/server_results/results/filter"),
]

REPEAT_TWICE = {"svd", "pca", "music", "esprit", "fastica", "nmf", "particle_1k"}
CSV_FIELDS = ["algorithm", "category", "N", "B", "variant", "gpus", "repeat_idx",
              "n_kernels", "n_memcpy", "n_memset", "n_cuda_events",
              "model_seq_steps", "model_fused_steps", "outer_iters", "kpi_table",
              "kpi_census", "ratio_census_over_model", "wall_ms", "top_kernels"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# =========================================================================
# CONFIGURATION LIST (union of both GPUs' benchmarked configurations)
# =========================================================================

def config_list():
    """Return dict (alg, N, B, variant) -> sorted list of GPU labels, plus input hashes."""
    configs = defaultdict(set)
    hashes = {}
    for label, csv_rel, has_ta, iir_dir in GPU_SOURCES:
        csv_path = BASE / csv_rel
        if not csv_path.exists():
            print(f"  WARNING: {csv_path} missing; skipping {label}")
            continue
        hashes[Path(csv_rel).as_posix()] = sha256_file(csv_path)
        points = ar.load_csv(str(csv_path), gpu_name=label)
        if iir_dir:
            for jf in sorted((BASE / iir_dir).glob("iir_butter4_*.json")):
                points.append(ar.load_json(str(jf)))
        for p in points:
            if p.algorithm == "iir_butter4":
                variant = "torchaudio" if has_ta else "fallback"
            else:
                variant = "default"
            configs[(p.algorithm, int(p.signal_length), int(p.batch_size), variant)].add(label)
    return {k: sorted(v) for k, v in configs.items()}, hashes


# =========================================================================
# PROFILING
# =========================================================================

def profiler_self_check(tries=3):
    """Run up to `tries` profiling sessions (the first session in a process
    sometimes drops CUDA activity). Return the number of CUDA kernels seen in
    the first session that sees any; 0 if none did. Prints diagnostics on failure."""
    x = torch.ones(1 << 16, device="cuda")
    torch.cuda.synchronize()
    last = None
    for t in range(tries):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            y = x * 2.0 + 1.0
            torch.cuda.synchronize()
        n = 0
        for ka in prof.key_averages():
            if ka.device_type == DeviceType.CUDA:
                n += ka.count
        last = prof
        if n >= 1:
            if t > 0:
                print(f"  (profiler needed {t + 1} sessions before CUDA events appeared)")
            del y
            return n
    del y
    try:
        print("  supported_activities:", torch.profiler.supported_activities())
    except Exception as e:  # pragma: no cover
        print("  supported_activities: unavailable", e)
    try:
        print("  events seen in the last session (first 15 rows):")
        print(last.key_averages().table(sort_by="count", row_limit=15))
    except Exception as e:  # pragma: no cover
        print("  (could not print table)", e)
    return 0


def count_launches(alg, N, B, variant, repeat, warmup=3):
    """Set up the algorithm exactly as the harness does, warm up, then profile
    `repeat` single invocations. Returns list of dicts (one per repeat)."""
    setup_fn, run_fn, defaults = ALL_ALGORITHMS[alg]
    device = torch.device("cuda")
    state = setup_fn(signal_length=N, batch_size=B, precision="fp32",
                     device=device, **defaults)
    note = ""
    if alg == "iir_butter4":
        if variant == "fallback":
            state["_lfilter"] = None          # force the pure-torch Python loop
        elif state.get("_lfilter") is None:
            note = "torchaudio unavailable locally; torchaudio variant not measurable"
            return [], note
    for _ in range(warmup):
        run_fn(state)
    torch.cuda.synchronize()
    out = []
    for r in range(repeat):
        t0 = time.perf_counter()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=False, profile_memory=False, with_stack=False) as prof:
            run_fn(state)
            torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - t0) * 1e3
        names = Counter()
        n_memcpy = n_memset = 0
        for ka in prof.key_averages():
            if ka.device_type != DeviceType.CUDA:
                continue
            key = ka.key
            if key.startswith("Memcpy"):
                n_memcpy += ka.count
            elif key.startswith("Memset"):
                n_memset += ka.count
            else:
                names[key] += ka.count
        out.append({"n_kernels": int(sum(names.values())), "n_memcpy": int(n_memcpy),
                    "n_memset": int(n_memset),
                    "n_cuda_events": int(sum(names.values()) + n_memcpy + n_memset),
                    "wall_ms": wall_ms,
                    "top_kernels": names.most_common(8)})
        del prof
    del state
    torch.cuda.empty_cache()
    return out, note


def model_counts(alg, N, B, variant):
    has_ta = (variant != "fallback")
    seq = int(get_seq_steps(alg, N, B, has_torchaudio=has_ta))
    fused = int(get_fused_steps(alg, N, B))
    outer = 0
    if alg in SEQ_STEPS:
        try:
            outer = int(SEQ_STEPS[alg](N, B, has_torchaudio=has_ta))
        except TypeError:
            outer = int(SEQ_STEPS[alg](N, B))
    kpi_table = KERNELS_PER_ITER.get(alg, 0) if outer > 0 else 0
    return seq, fused, outer, kpi_table



# =========================================================================
# NSYS BACKEND
# =========================================================================

NSYS_CANDIDATES = [
    r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2023.4.4\target-windows-x64\nsys.exe",
]


def find_nsys(explicit=None):
    if explicit:
        return explicit
    p = shutil.which("nsys")
    if p:
        return p
    for c in NSYS_CANDIDATES:
        if Path(c).exists():
            return c
    hits = sorted(glob.glob(r"C:\Program Files\NVIDIA Corporation\Nsight Systems*\target-windows-x64\nsys.exe"))
    return hits[-1] if hits else None


def worker(alg, N, B, variant, marker=None, warmup=3):
    """Child-process body: set up like the harness, warm up, bracket one call.
    Writes a JSON marker file (status, wall_ms) because nsys on Windows does not
    relay the child's stdout to the parent process."""
    def _mark(status, wall_ms=None):
        if marker:
            with open(marker, "w", encoding="utf-8") as f:
                json.dump({"status": status, "wall_ms": wall_ms}, f)
    setup_fn, run_fn, defaults = ALL_ALGORITHMS[alg]
    device = torch.device("cuda")
    state = setup_fn(signal_length=N, batch_size=B, precision="fp32", device=device, **defaults)
    if alg == "iir_butter4":
        if variant == "fallback":
            state["_lfilter"] = None
        elif state.get("_lfilter") is None:
            print("WORKER_SKIP torchaudio unavailable locally", flush=True)
            _mark("skip")
            return 3
    for _ in range(warmup):
        run_fn(state)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    torch.cuda.cudart().cudaProfilerStart()
    run_fn(state)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    wall_ms = (time.perf_counter() - t0) * 1e3
    print(f"WORKER_OK wall_ms={wall_ms:.2f}", flush=True)
    _mark("ok", wall_ms)
    return 0


def _read_csv_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    # nsys may prepend non-CSV lines; start at the first line that looks like a header
    start = 0
    for i, ln in enumerate(lines):
        if ln.startswith('"') or ln.lower().startswith("time"):
            start = i
            break
    return list(csv.DictReader(lines[start:]))


def count_launches_nsys(alg, N, B, variant, repeat, nsys_path, keep=False):
    tmp_dir = OUT_DIR / "nsys_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out = []
    note = ""
    for r in range(repeat):
        stem = f"{alg}_N{N}_B{B}_{variant}_r{r}"
        rep = tmp_dir / f"{stem}.nsys-rep"
        stats_base = tmp_dir / f"{stem}_stats"
        marker = tmp_dir / f"{stem}.worker.json"
        for old in glob.glob(str(tmp_dir / f"{stem}*")):
            try:
                os.remove(old)
            except OSError:
                pass
        cmd = [nsys_path, "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none",
               "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
               "--force-overwrite=true", "--output", str(rep),
               sys.executable, str(Path(__file__).resolve()), "--worker", alg, str(N), str(B), variant,
               "--marker", str(marker)]
        timeout = 3600 if (alg == "iir_butter4" and variant == "fallback") else 900
        pr = subprocess.run(cmd, cwd=str(BASE), capture_output=True, text=True, timeout=timeout)
        stdout = pr.stdout or ""
        status, wall_ms = None, float("nan")
        if marker.exists():
            try:
                with open(marker, encoding="utf-8") as f:
                    m = json.load(f)
                status = m.get("status")
                wall_ms = float(m.get("wall_ms") or float("nan"))
            except Exception:
                status = None
        if status is None:  # fall back to stdout if it was relayed
            if "WORKER_SKIP" in stdout:
                status = "skip"
            elif "WORKER_OK" in stdout:
                status = "ok"
                for tok in stdout.split():
                    if tok.startswith("wall_ms="):
                        wall_ms = float(tok.split("=", 1)[1])
        if status == "skip":
            note = "torchaudio unavailable locally; torchaudio variant not measurable"
            return [], note
        if status != "ok" or not rep.exists():
            tail = (pr.stderr or "")[-600:] + (stdout[-600:])
            raise RuntimeError(f"nsys profile failed (rc={pr.returncode}, marker={'yes' if marker.exists() else 'no'}, "
                               f"report={'yes' if rep.exists() else 'no'}): {tail}")
        cmd2 = [nsys_path, "stats", "--report", "cuda_gpu_kern_sum", "--report", "cuda_gpu_mem_time_sum",
                "--format", "csv", "--force-export=true", "--output", str(stats_base), str(rep)]
        pr2 = subprocess.run(cmd2, cwd=str(BASE), capture_output=True, text=True, timeout=1800)
        kern_csv = Path(str(stats_base) + "_cuda_gpu_kern_sum.csv")
        mem_csv = Path(str(stats_base) + "_cuda_gpu_mem_time_sum.csv")
        names = Counter()
        if kern_csv.exists():
            for row in _read_csv_rows(kern_csv):
                inst = row.get("Instances") or row.get("Count") or "0"
                names[row.get("Name", "?")] += int(float(inst.replace(",", "")))
        elif status == "ok":
            # no kernels file: either zero kernels or stats failure
            if pr2.returncode != 0:
                raise RuntimeError(f"nsys stats failed (rc={pr2.returncode}): {(pr2.stderr or '')[-600:]}")
        n_memcpy = n_memset = 0
        if mem_csv.exists():
            for row in _read_csv_rows(mem_csv):
                op = (row.get("Operation") or row.get("Name") or "").lower()
                cnt = int(float((row.get("Count") or row.get("Instances") or "0").replace(",", "")))
                if "memcpy" in op:
                    n_memcpy += cnt
                elif "memset" in op:
                    n_memset += cnt
        out.append({"n_kernels": int(sum(names.values())), "n_memcpy": int(n_memcpy),
                    "n_memset": int(n_memset),
                    "n_cuda_events": int(sum(names.values()) + n_memcpy + n_memset),
                    "wall_ms": wall_ms, "top_kernels": names.most_common(8)})
        if not keep:
            for old in glob.glob(str(tmp_dir / f"{stem}*")):
                try:
                    os.remove(old)
                except OSError:
                    pass
    return out, note


# =========================================================================
# CSV / JSON PERSISTENCE
# =========================================================================

def read_done():
    done = {}
    if not CSV_OUT.exists():
        return done
    with open(CSV_OUT, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["algorithm"], int(row["N"]), int(row["B"]), row["variant"], int(row["repeat_idx"]))
            done[key] = row
    return done


def append_row(row):
    new = not CSV_OUT.exists()
    with open(CSV_OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


# =========================================================================
# SUMMARY
# =========================================================================

def print_summary(rows):
    """rows: list of CSV dict rows (strings)."""
    def f(x, default=float("nan")):
        try:
            return float(x)
        except Exception:
            return default

    print("\n" + "=" * 92)
    print("  A. PYTHON-LOOP AND FALLBACK ALGORITHMS: census launches vs model S_o")
    print("=" * 92)
    print(f"  {'algorithm':14s} {'variant':10s} {'N':>6s} {'B':>5s} {'census':>8s} {'model S_o':>9s} "
          f"{'ratio':>6s} {'outer':>6s} {'kpi table':>9s} {'kpi census':>10s} {'memcpy':>6s} gpus")
    seq_rows = [r for r in rows if f(r["model_seq_steps"]) > 0 or r["variant"] == "fallback"]
    seq_rows.sort(key=lambda r: (r["algorithm"], r["variant"], int(r["N"]), int(r["B"]), int(r["repeat_idx"])))
    per_alg_kpi = defaultdict(list)
    for r in seq_rows:
        if int(r["repeat_idx"]) != 0:
            continue
        ratio = f(r["ratio_census_over_model"])
        print(f"  {r['algorithm']:14s} {r['variant']:10s} {int(r['N']):6d} {int(r['B']):5d} "
              f"{int(f(r['n_kernels'])):8d} {int(f(r['model_seq_steps'])):9d} {ratio:6.2f} "
              f"{int(f(r['outer_iters'])):6d} {f(r['kpi_table']):9.1f} {f(r['kpi_census']):10.1f} "
              f"{int(f(r['n_memcpy'])):6d} {r['gpus']}")
        per_alg_kpi[(r["algorithm"], r["variant"])].append(f(r["kpi_census"]))
    print("\n  Kernels per outer iteration, table vs census (min..max over configs):")
    for (alg, var), vals in sorted(per_alg_kpi.items()):
        tab = KERNELS_PER_ITER.get(alg, 0)
        vals = [v for v in vals if v == v]
        if vals:
            print(f"    {alg:14s} {var:10s} table {tab:5.1f}   census {min(vals):6.1f} .. {max(vals):6.1f}"
                  f"   ratio {min(vals)/tab if tab else float('nan'):5.2f} .. {max(vals)/tab if tab else float('nan'):5.2f}")

    print("\n" + "=" * 92)
    print("  B. PARALLEL AND FUSED ALGORITHMS: kernels per invocation (model S_o = 0)")
    print("=" * 92)
    par_rows = [r for r in rows if f(r["model_seq_steps"]) == 0 and r["variant"] != "fallback"]
    by_alg = defaultdict(list)
    for r in par_rows:
        by_alg[r["algorithm"]].append(r)
    print(f"  {'algorithm':22s} {'cfgs':>4s} {'kernels min..max':>18s} {'memcpy':>6s} {'memset':>6s} {'fused steps':>11s}  flag")
    for alg in sorted(by_alg, key=lambda a: (CATEGORY_OF.get(a, ""), a)):
        rs = by_alg[alg]
        n_cfg = len({(int(r["N"]), int(r["B"])) for r in rs})
        ks = [int(f(r["n_kernels"])) for r in rs]
        mc = max(int(f(r["n_memcpy"])) for r in rs)
        ms = max(int(f(r["n_memset"])) for r in rs)
        fs = max(int(f(r["model_fused_steps"])) for r in rs)
        flag = "LAUNCH-HEAVY (>= 20)" if max(ks) >= 20 else ""
        print(f"  {alg:22s} {n_cfg:4d} {min(ks):8d} .. {max(ks):5d} {mc:6d} {ms:6d} {fs:11d}  {flag}")
    print("\n  Per-configuration detail for launch-heavy and repeated algorithms:")
    for alg in sorted(by_alg):
        rs = sorted(by_alg[alg], key=lambda r: (int(r["N"]), int(r["B"]), int(r["repeat_idx"])))
        if max(int(f(r["n_kernels"])) for r in rs) < 20 and alg not in REPEAT_TWICE:
            continue
        for r in rs:
            print(f"    {alg:22s} N={int(r['N']):6d} B={int(r['B']):5d} rep={r['repeat_idx']} "
                  f"kernels={int(f(r['n_kernels'])):6d} memcpy={int(f(r['n_memcpy'])):4d} "
                  f"wall={f(r['wall_ms']):8.1f} ms  top: {r['top_kernels'][:110]}")


# =========================================================================
# MAIN
# =========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algs", nargs="*", default=None, help="restrict to these algorithms")
    ap.add_argument("--repeat", type=int, default=None, help="override repeat count for all")
    ap.add_argument("--force", action="store_true", help="re-profile configurations already in the CSV")
    ap.add_argument("--summary-only", action="store_true", help="only print tables from the CSV")
    ap.add_argument("--skip", nargs="*", default=[], help="config keys to skip, e.g. iir_butter4:16384:1:fallback")
    ap.add_argument("--backend", choices=["auto", "torch", "nsys"], default="auto")
    ap.add_argument("--nsys", default=None, help="path to nsys executable")
    ap.add_argument("--keep-reports", action="store_true", help="keep .nsys-rep and stats CSVs")
    ap.add_argument("--worker", nargs=4, metavar=("ALG", "N", "B", "VARIANT"), default=None,
                    help="internal: run one configuration under nsys and exit")
    ap.add_argument("--marker", default=None, help="internal: marker file written by the worker")
    args = ap.parse_args()

    if args.worker:
        alg, N, B, variant = args.worker[0], int(args.worker[1]), int(args.worker[2]), args.worker[3]
        sys.exit(worker(alg, N, B, variant, marker=args.marker))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configs, hashes = config_list()
    if args.summary_only:
        print_summary(list(read_done().values()))
        return

    if not torch.cuda.is_available():
        print("CUDA not available; aborting")
        sys.exit(1)
    dev = torch.cuda.get_device_name(0)
    print(f"TOMLSignals EXP-CR-005 kernel-launch census on {dev}")
    print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, python {sys.version.split()[0]}")
    backend = args.backend
    nsys_path = None
    if backend in ("auto", "torch"):
        if backend == "torch":
            n_check = profiler_self_check()
        else:
            n_check = 1 if ProfilerActivity.CUDA in torch.profiler.supported_activities() else 0
            if n_check:
                n_check = profiler_self_check()
        if n_check >= 1:
            backend = "torch"
        elif args.backend == "torch":
            print("ERROR: torch.profiler has no CUDA activity on this build; use --backend nsys")
            sys.exit(2)
        else:
            backend = "nsys"
    if backend == "nsys":
        nsys_path = find_nsys(args.nsys)
        if not nsys_path:
            print("ERROR: nsys not found; pass --nsys <path to nsys.exe>")
            sys.exit(2)
        ver = subprocess.run([nsys_path, "--version"], capture_output=True, text=True, timeout=60)
        print(f"backend: nsys ({nsys_path}) {ver.stdout.strip()}")
    else:
        print("backend: torch.profiler")

    done = read_done()
    keys = sorted(configs.keys(), key=lambda k: (CATEGORY_OF.get(k[0], ""), k[0], k[3], k[1], k[2]))
    if args.algs:
        keys = [k for k in keys if k[0] in args.algs]
    skip = set(args.skip)
    print(f"{len(keys)} configurations to census; {len(done)} rows already in CSV")

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    n_new = 0
    for i, (alg, N, B, variant) in enumerate(keys, 1):
        key_str = f"{alg}:{N}:{B}:{variant}"
        if key_str in skip:
            print(f"[{i}/{len(keys)}] {key_str} skipped by request")
            continue
        repeat = args.repeat or (2 if alg in REPEAT_TWICE else 1)
        if not args.force and all((alg, N, B, variant, r) in done for r in range(repeat)):
            continue
        print(f"[{i}/{len(keys)}] {alg} N={N} B={B} {variant} ...", end="", flush=True)
        try:
            if backend == "nsys":
                results, note = count_launches_nsys(alg, N, B, variant, repeat, nsys_path, keep=args.keep_reports)
            else:
                results, note = count_launches(alg, N, B, variant, repeat)
        except Exception as e:  # keep going; record the failure
            print(f" ERROR {type(e).__name__}: {e}")
            torch.cuda.empty_cache()
            continue
        if not results:
            print(f" not measured ({note})")
            continue
        seq, fused, outer, kpi_table = model_counts(alg, N, B, variant)
        for r_idx, res in enumerate(results):
            kpi_census = res["n_kernels"] / outer if outer > 0 else float("nan")
            ratio = res["n_kernels"] / seq if seq > 0 else float("nan")
            row = {"algorithm": alg, "category": CATEGORY_OF.get(alg, ""), "N": N, "B": B,
                   "variant": variant, "gpus": "+".join(configs[(alg, N, B, variant)]),
                   "repeat_idx": r_idx, "n_kernels": res["n_kernels"], "n_memcpy": res["n_memcpy"],
                   "n_memset": res["n_memset"], "n_cuda_events": res["n_cuda_events"],
                   "model_seq_steps": seq, "model_fused_steps": fused, "outer_iters": outer,
                   "kpi_table": kpi_table, "kpi_census": f"{kpi_census:.3f}" if kpi_census == kpi_census else "",
                   "ratio_census_over_model": f"{ratio:.4f}" if ratio == ratio else "",
                   "wall_ms": f"{res['wall_ms']:.2f}",
                   "top_kernels": " | ".join(f"{n}x {k[:60]}" for k, n in res["top_kernels"])}
            append_row(row)
            done[(alg, N, B, variant, r_idx)] = row
            n_new += 1
        k0 = results[0]["n_kernels"]
        extra = f" (S_o model {seq}, ratio {k0/seq:.2f})" if seq > 0 else ""
        print(f" kernels={k0} memcpy={results[0]['n_memcpy']}{extra}  {results[0]['wall_ms']:.0f} ms")

    rows = list(read_done().values())
    print_summary(rows)
    summary = {"experiment": "EXP-CR-005", "started_utc": started, "device": dev,
               "backend": backend, "nsys": nsys_path,
               "torch": torch.__version__, "cuda": torch.version.cuda,
               "python": sys.version.split()[0], "platform": platform.platform(),
               "input_hashes": hashes, "n_configurations": len(keys), "n_rows": len(rows),
               "kernels_per_iter_table": KERNELS_PER_ITER, "rows": rows}
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(BASE), capture_output=True, text=True, timeout=10)
        summary["git_commit"] = sha.stdout.strip() if sha.returncode == 0 else "unavailable"
    except Exception:  # pragma: no cover
        summary["git_commit"] = "unavailable"
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  {n_new} new rows this run; CSV: {CSV_OUT.relative_to(BASE).as_posix()}; "
          f"JSON: {JSON_OUT.relative_to(BASE).as_posix()}")


if __name__ == "__main__":
    main()
