#!/usr/bin/env python3
"""Scan models/ -> deduplicate checkpoints -> emit model_registry.json.

Dedup rules (see DECISIONS.md):
  * classical .joblib : identity = (family, variant, horizon, group, algo);
                        keep the LAST checkpoint = max(epochs, patience, lag).
  * deep .pt          : identity = (family, [symbol], horizon, group, model);
                        keep the BEST checkpoint = min(rmse) from embedded metrics
                        (this mirrors how the training sweep itself selected "best").

Run:  python build_registry.py --models-dir ../models --out app/model_registry.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.features.groups import basis_for, expected_columns  # noqa: E402

ALGOS = ["elastic_net", "ridge", "random_forest", "hist_gbr", "mlp"]
DEEP_MODELS = [
    "GRU", "LSTM", "Informer", "PatchTST", "TFT",
    "FEDformer", "Autoformer", "NBEATS", "NHITS", "MambaStock",
]


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _mk_id(parts: list[str]) -> str:
    """Stable, collision-safe id: slug each part, join with '__'."""
    return "__".join(_slug(p) for p in parts if p)


def _find(tokens: list[str], name: str) -> str | None:
    """Return the token that appears in name, longest first (avoids 'Mamba' vs 'MambaStock')."""
    for tok in sorted(tokens, key=len, reverse=True):
        if re.search(rf"(?:^|_){re.escape(tok)}(?:_|$|\.)", name):
            return tok
    return None


def _group(name: str) -> str | None:
    m = re.search(r"(G[1-5])_", name)
    return m.group(1) if m else None


def _ints(name: str) -> tuple[int, int, int]:
    def g(pat: str) -> int:
        m = re.search(pat, name)
        return int(m.group(1)) if m else 0
    return g(r"_e(\d+)"), g(r"_p(\d+)"), g(r"lag(\d+)")


def _horizon(path: Path, name: str) -> str:
    return "h5" if ("target_h5" in str(path)) or ("_h5" in name and "h5" in name) else "h1"


def _variant(path: Path) -> str | None:
    m = re.search(r"preprocessing_variants[\\/]+([^\\/]+)", str(path))
    return m.group(1) if m else None


def scan_classical(models_dir: Path) -> dict[tuple, dict]:
    import joblib

    best: dict[tuple, dict] = {}
    for f in models_dir.rglob("*.joblib"):
        rel = f.relative_to(models_dir)
        family = rel.parts[0]
        name = f.stem
        group = _group(name)
        algo = _find(ALGOS, name)
        if group is None or algo is None:
            continue
        horizon = "h5" if "target_h5" in name or "target_h5" in str(rel) else "h1"
        variant = _variant(f)
        e, p, lag = _ints(name)
        key = (family, variant, horizon, group, algo)
        cand = {"e": e, "p": p, "lag": lag, "path": f}
        prev = best.get(key)
        if prev is None or (cand["e"], cand["p"], cand["lag"]) > (prev["e"], prev["p"], prev["lag"]):
            best[key] = cand

    out: dict[tuple, dict] = {}
    for (family, variant, horizon, group, algo), c in best.items():
        f = c["path"]
        # Authoritative ordered feature names live inside the fitted pipeline.
        feature_names = None
        try:
            pipe = joblib.load(f)
            fn = getattr(pipe, "feature_names_in_", None)
            if fn is not None:
                feature_names = list(fn)
        except Exception:
            feature_names = None
        if feature_names is None:
            feature_names = expected_columns(family, group)

        parts = [family]
        if variant:
            parts.append(variant)
        parts += [horizon, group, algo]
        model_id = _mk_id(parts)
        out[model_id] = {
            "model_id": model_id,
            "kind": "classical",
            "family": family,
            "variant": variant,
            "horizon": horizon,
            "group": group,
            "algo": algo,
            "model_name": algo,
            "symbol": None,
            "basis": basis_for(family),
            "feature_names": feature_names,
            "n_features": len(feature_names),
            "seq_len": 1,
            "metrics": {},
            "path": str(f.relative_to(models_dir)).replace("\\", "/"),
        }
    return out


def scan_deep(models_dir: Path) -> dict[tuple, dict]:
    import torch

    best: dict[tuple, dict] = {}
    pts = list(models_dir.rglob("*.pt"))
    for i, f in enumerate(pts, 1):
        rel = f.relative_to(models_dir)
        family = rel.parts[0]
        name = f.stem
        per_symbol = family == "nasdaq100_best_deep_per_symbol"
        symbol = rel.parts[2] if per_symbol and len(rel.parts) > 2 else None
        try:
            ck = torch.load(f, map_location="cpu", weights_only=False)
        except Exception as exc:
            print(f"  [warn] skip {rel}: {exc}")
            continue
        model_name = ck.get("model_name") or _find(DEEP_MODELS, name)
        group = _group(ck.get("feature_group", "")) or _group(name)
        if model_name is None or group is None:
            continue
        horizon = "h5" if ck.get("target") == "target_h5" or "target_h5" in str(rel) else "h1"
        rmse = float(ck.get("metrics", {}).get("rmse", float("inf")))
        key = (family, symbol, horizon, group, model_name)
        cand = {
            "rmse": rmse,
            "path": f,
            "n_features": int(ck.get("n_features", 0)),
            "seq_len": int(ck.get("seq_len", ck.get("config", {}).get("lag_window", 0))),
            "config": ck.get("config", {}),
            "metrics": ck.get("metrics", {}),
        }
        prev = best.get(key)
        if prev is None or cand["rmse"] < prev["rmse"]:
            best[key] = cand
        if i % 400 == 0:
            print(f"  ...scanned {i}/{len(pts)} .pt files")

    out: dict[tuple, dict] = {}
    for (family, symbol, horizon, group, model_name), c in best.items():
        parts = [family]
        if symbol:
            parts.append(symbol)
        parts += [horizon, group, model_name]
        model_id = _mk_id(parts)
        out[model_id] = {
            "model_id": model_id,
            "kind": "deep",
            "family": family,
            "variant": None,
            "horizon": horizon,
            "group": group,
            "algo": model_name,
            "model_name": model_name,
            "symbol": symbol,
            "basis": basis_for(family),
            "feature_names": expected_columns(family, group),
            "n_features": c["n_features"],
            "seq_len": c["seq_len"],
            "config": c["config"],
            "metrics": {k: (None if v != v else v) for k, v in c["metrics"].items()},  # NaN->None
            "path": str(c["path"].relative_to(models_dir)).replace("\\", "/"),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", type=Path, default=Path(__file__).resolve().parents[1] / "src" / "artifacts")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "app" / "model_registry.json")
    args = ap.parse_args()

    models_dir = args.models_dir.resolve()
    if not models_dir.exists():
        raise SystemExit(f"models dir not found: {models_dir}")

    print(f"Scanning classical models in {models_dir} ...")
    classical = scan_classical(models_dir)
    print(f"  classical (deduped): {len(classical)}")

    print("Scanning deep models ...")
    deep = scan_deep(models_dir)
    print(f"  deep (deduped): {len(deep)}")

    registry = {**classical, **deep}

    # Summary counts for convenience.
    by_family: dict[str, int] = {}
    for e in registry.values():
        by_family[e["family"]] = by_family.get(e["family"], 0) + 1

    payload = {
        "models_dir": str(models_dir).replace("\\", "/"),
        "count": len(registry),
        "by_family": dict(sorted(by_family.items())),
        "models": registry,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {len(registry)} models -> {args.out}")
    for fam, n in sorted(by_family.items()):
        print(f"  {fam}: {n}")


if __name__ == "__main__":
    main()
