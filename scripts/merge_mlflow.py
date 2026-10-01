"""Merge the MLflow runs of extra tracking databases into one database (default: mlflow.db).

The final specialists / soft MoE and the second Task 4 recipe were trained in separate Kaggle sessions, each with its
own tracking database (mlflow_task23_v2.db, mlflow_task4_v2.db). This copies their experiments, runs, parameters,
tags and full metric histories into mlflow.db so one MLflow UI shows the whole project. Run it once; it is idempotent
(runs that already exist with the same name and start time are skipped). Artifacts (images) are not copied.

  python scripts/merge_mlflow.py --dst mlflow.db --src mlflow_task23_v2.db mlflow_task4_v2.db
"""
import argparse
import shutil
from pathlib import Path

from mlflow.entities import Metric, Param, RunTag
from mlflow.tracking import MlflowClient

ap = argparse.ArgumentParser()
ap.add_argument("--dst", default="mlflow.db")
ap.add_argument("--src", nargs="+", required=True)
a = ap.parse_args()
uri = lambda p: f"sqlite:///{Path(p).resolve().as_posix()}"
shutil.copy(a.dst, a.dst + ".bak")
dst = MlflowClient(uri(a.dst))
SKIP_TAGS = {"mlflow.runName", "mlflow.user", "mlflow.source.name", "mlflow.source.type", "mlflow.source.git.commit"}


def chunks(items, n):
    for i in range(0, len(items), n):
        yield items[i:i + n]


copied = skipped = 0
for src_path in a.src:
    src = MlflowClient(uri(src_path))
    for exp in src.search_experiments():
        if exp.name == "Default":
            continue
        d = dst.get_experiment_by_name(exp.name)
        dst_id = d.experiment_id if d else dst.create_experiment(exp.name)
        existing = {(r.info.run_name, r.info.start_time) for r in dst.search_runs([dst_id], max_results=10000)}
        id_map = {}
        for r in src.search_runs([exp.experiment_id], max_results=10000, order_by=["attributes.start_time ASC"]):
            key = (r.info.run_name, r.info.start_time)
            if key in existing:
                skipped += 1
                continue
            tags = {k: v for k, v in r.data.tags.items() if k not in SKIP_TAGS}
            parent = tags.get("mlflow.parentRunId")
            if parent:
                tags["mlflow.parentRunId"] = id_map.get(parent, parent)
            new = dst.create_run(dst_id, start_time=r.info.start_time, tags=tags, run_name=r.info.run_name)
            id_map[r.info.run_id] = new.info.run_id
            metrics = [Metric(m.key, m.value, m.timestamp, m.step) for k in r.data.metrics for m in src.get_metric_history(r.info.run_id, k)]
            for c in chunks(metrics, 500):
                dst.log_batch(new.info.run_id, metrics=c)
            for c in chunks([Param(k, str(v)) for k, v in r.data.params.items()], 90):
                dst.log_batch(new.info.run_id, params=c)
            dst.set_terminated(new.info.run_id, status=r.info.status, end_time=r.info.end_time)
            copied += 1
print(f"copied {copied} runs, skipped {skipped} already present; backup at {a.dst}.bak")
