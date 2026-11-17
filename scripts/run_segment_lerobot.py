#!/usr/bin/env python3
"""Submit LeRobot front_head segmentation and table/floor mask repair to Slurm."""

import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

from segment_lerobot import MASK_NAMES, discover_lerobot, load_json, mask_relative, write_json


REPO = Path(__file__).resolve().parents[1]
SEGMENT = REPO / "scripts/segment_lerobot.py"
REPAIR = REPO / "scripts/repair_lerobot_background_masks.py"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--manifest", type=Path, help="LeRobot v2.1 dataset/prepared/manifest.json.")
    source.add_argument("--reuse-segmentation", type=Path,
                        help="Completed SAM segmentation directory with plan.json and completed.json.")
    parser.add_argument("--output", type=Path, required=True, help="New run directory, or report directory with --no-media.")
    parser.add_argument("--segmentation-only", action="store_true", help="Publish the first-pass masks only.")
    parser.add_argument("--no-media", action="store_true", help="Write statistics and mask Parquets without media.")
    parser.add_argument("--dataset-root", type=Path, help="LeRobot root to scan, or output root for an external manifest.")
    parser.add_argument("--ordinals", type=int, nargs="+", help="Zero-based input episode positions for a small run.")
    parser.add_argument("--shards", type=int, default=8, help="One GPU per shard (default: 8).")
    parser.add_argument("--partition", default="wangdonglinlab", help="Slurm partition (default: wangdonglinlab).")
    parser.add_argument("--account", default="wangdonglin", help="Slurm account (default: wangdonglin).")
    parser.add_argument("--nodelist", help="Optional Slurm node list.")
    parser.add_argument("--time", default="1-00:00:00", help="Time limit for each stage (default: 1 day).")
    parser.add_argument("--python", type=Path, default=REPO.parent / "conda_envs/sam31/bin/python",
                        help="Python executable in the SAM 3.1 environment.")
    parser.add_argument("--deps", type=Path, help="Optional Python dependency directory containing PyAV and PyArrow.")
    parser.add_argument("--checkpoint", type=Path, default=REPO.parent / "weight/sam3.1_multiplex.pt",
                        help="Path to the local SAM 3.1 checkpoint.")
    args = parser.parse_args()
    if not (args.manifest or args.reuse_segmentation or args.dataset_root):
        parser.error("Provide --dataset-root, --manifest, or --reuse-segmentation.")
    if args.shards < 1:
        parser.error("--shards must be positive.")
    if args.reuse_segmentation and (args.ordinals or args.segmentation_only):
        parser.error("--ordinals and --segmentation-only apply to a new segmentation.")
    if args.dataset_root and not args.no_media:
        parser.error("--dataset-root requires --no-media.")
    return args


def submit(args, sbatch, python, python_path, stage, worker, logs, shards, mem, dependency=None):
    logs.mkdir(exist_ok=True)
    command = [
        sbatch, "--parsable", "--requeue", f"--partition={args.partition}", f"--account={args.account}",
        f"--job-name=sam31-lerobot-{stage}", f"--array=0-{shards - 1}", "--nodes=1", "--ntasks=1",
        "--gpus=1", "--cpus-per-task=8", f"--mem={mem}", f"--time={args.time}",
        f"--output={logs / (stage + '-%A_%a.log')}",
    ]
    if args.nodelist:
        command.append(f"--nodelist={args.nodelist}")
    if dependency:
        command.extend((f"--dependency=afterok:{dependency}", "--kill-on-invalid-dep=yes"))
    environment = ["env", f"PYTHONPATH={python_path}", "HF_HUB_OFFLINE=1",
                   "OMP_NUM_THREADS=2", "OPENBLAS_NUM_THREADS=1", "TQDM_DISABLE=1"]
    wrapped = "exec " + shlex.join([*environment, str(python), "-Bu", *map(str, worker)])
    wrapped += ' --shard "$SLURM_ARRAY_TASK_ID"'
    result = subprocess.check_output([*command, "--wrap", wrapped], text=True).strip()
    job = result.split(";", 1)[0]
    if not job.isdecimal():
        raise RuntimeError(f"Unexpected sbatch response: {result!r}")
    return job


def prepare_reuse(segmentation, output, shards):
    done = load_json(segmentation / "completed.json")
    plan = load_json(segmentation / "plan.json")
    tasks = plan["tasks"]
    manifest = Path(done["training_manifest"])
    entries = load_json(manifest)["episodes"]
    if len(tasks) != len(entries) or len(tasks) != done["episodes"] or not tasks:
        raise ValueError(f"Segmentation completion marker does not match its plan: {segmentation}")
    for task, entry in zip(tasks, entries, strict=True):
        if (task["video"] != entry["videos"]["front_head"]
                or task["mask_parquet"] != entry["background_masks_parquet"]
                or not Path(task["mask_parquet"]).is_file()
                or not Path(task["video"]).is_file()):
            raise ValueError(f"Inconsistent source artifacts: {task['name']}")
        episode = segmentation / task["name"]
        artifacts = ([episode / "labels.npz"] if plan.get("no_media") else
                     [episode / "front_head" / "masks" / f"{name}.mkv" for name in MASK_NAMES])
        if not all(artifact.is_file() for artifact in artifacts):
            raise ValueError(f"Missing segmentation masks: {episode}")
    loads = [0] * min(shards, len(tasks))
    for task in sorted(tasks, key=lambda t: t["frames"], reverse=True):
        shard = loads.index(min(loads))
        task["shard"] = shard
        loads[shard] += task["frames"]
    plan["loads"] = loads
    path = output / "repair_plan.json"
    write_json(path, plan)
    root = Path(plan.get("dataset_root") or Path(plan["manifest"]).parent.parent)
    raw_masks = Path(tasks[0]["mask_parquet"])
    # Recover the original mask root even when the selection contains only one episode.
    relative = mask_relative(tasks[0], root)
    for _ in relative.parts:
        raw_masks = raw_masks.parent
    if any(Path(task["mask_parquet"]) != raw_masks / mask_relative(task, root)
           for task in tasks):
        raise ValueError(f"Segmentation paths do not share the same mask root: {segmentation}")
    return path, manifest, raw_masks, len(loads)


def main():
    args = parse_args()
    python = args.python.expanduser().absolute()
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    sbatch = shutil.which("sbatch") or "/soft/slurm/bin/sbatch"
    for executable in (python, Path(sbatch)):
        if not os.access(executable, os.X_OK):
            raise FileNotFoundError(executable)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not args.no_media and not all(shutil.which(tool) for tool in ("ffmpeg", "ffprobe")):
        raise RuntimeError("FFmpeg and ffprobe must be available in PATH.")
    dependencies = [str(REPO)]
    if args.deps:
        dependencies.insert(0, str(args.deps.expanduser().resolve(strict=True)))
    if os.environ.get("PYTHONPATH"):
        dependencies.append(os.environ["PYTHONPATH"])
    python_path = os.pathsep.join(dependencies)
    subprocess.run(
        [python, "-c", "import av; import pyarrow.parquet"],
        env={**os.environ, "PYTHONPATH": python_path}, check=True,
    )
    direct_scan = not (args.manifest or args.reuse_segmentation)
    if args.no_media:
        if direct_scan:
            inferred_root = args.dataset_root
        elif args.reuse_segmentation:
            prior = load_json(args.reuse_segmentation.expanduser() / "plan.json")
            source_manifest = Path(prior["manifest"])
            inferred_root = prior.get("dataset_root") or source_manifest.parent.parent
            if prior.get("direct_scan") and (args.dataset_root or Path(inferred_root)).resolve() != Path(inferred_root):
                raise ValueError("Reuse direct-scan segmentation with its original dataset root.")
        else:
            source_manifest = args.manifest.expanduser().resolve(strict=True)
            inferred_root = source_manifest.parent.parent
        dataset_root = Path(args.dataset_root or inferred_root).expanduser().resolve(strict=True)
        from publish_lerobot_masks import check_report_directory
        check_report_directory(output, dataset_root)
        report_directory = output
        output = Path(tempfile.mkdtemp(prefix=".sam31-masks-", dir=dataset_root))
        print(f"Run workspace: {output}", flush=True)
    else:
        output.mkdir(parents=True, exist_ok=False)
    logs = output / "logs"
    jobs = {}
    raw_masks = output / "background_masks"
    if args.reuse_segmentation:
        segmentation = args.reuse_segmentation.expanduser().resolve(strict=True)
        plan, raw_manifest, raw_masks, shards = prepare_reuse(segmentation, output, args.shards)
    else:
        if direct_scan:
            manifest = output / "source_episodes.json"
            discovered = discover_lerobot(dataset_root)
            write_json(manifest, discovered)
            entries = discovered["episodes"]
        else:
            manifest = args.manifest.expanduser().resolve(strict=True)
            entries = load_json(manifest)["episodes"]
        shards = min(args.shards, len(args.ordinals or entries))
        segmentation = output / "segmentation"
        raw_manifest = output / "manifest_background_masks.json"
        plan = segmentation / "plan.json"
        prepare = [
            python, "-Bu", SEGMENT, "prepare", "--manifest", manifest, "--output", segmentation,
            "--mask-root", raw_masks, "--shards", str(shards), "--enriched-manifest", raw_manifest,
        ]
        if args.no_media:
            prepare.extend(["--no-media", "--dataset-root", str(dataset_root)])
        if direct_scan:
            prepare.append("--direct-scan")
        if args.ordinals:
            prepare.extend(["--ordinals", *map(str, args.ordinals)])
        subprocess.run(prepare, env={**os.environ, "PYTHONPATH": python_path}, check=True)
        jobs["segmentation"] = submit(
            args, sbatch, python, python_path, "segment",
            [SEGMENT, "worker", "--output", segmentation, "--checkpoint", checkpoint],
            logs, shards, "24G",
        )
        print(f"Segmentation job: {jobs['segmentation']}", flush=True)
        write_json(output / "submission.json", dict(jobs=jobs, segmentation=str(segmentation),
                                                   training_manifest=str(raw_manifest)))
    if not args.segmentation_only:
        tracked_manifest = output / "manifest_background_masks_tracked.json"
        jobs["repair"] = submit(
            args, sbatch, python, python_path, "repair",
            [REPAIR, "--plan", plan, "--manifest", raw_manifest, "--masks-root", segmentation,
             "--original-root", raw_masks, "--repaired-root", output / "background_masks_tracked",
             "--output", output / "repair", "--enriched", tracked_manifest, "--checkpoint", checkpoint,
             *(["--five-masks"] if args.no_media else [])],
            logs, shards, "32G", dependency=jobs.get("segmentation"),
        )
        print(f"Table/floor repair job: {jobs['repair']}", flush=True)
        write_json(output / "submission.json", dict(jobs=jobs, segmentation=str(segmentation),
                                                   training_manifest=str(tracked_manifest)))
    if args.no_media:
        dependency = jobs.get("repair") or jobs["segmentation"]
        publisher = REPO / "scripts/publish_lerobot_masks.py"
        command = [python, "-Bu", publisher, "--work", output, "--dataset-root", dataset_root,
                   "--report-directory", report_directory]
        if args.segmentation_only:
            command.append("--segmentation-only")
        # A Slurm log held open on NFS prevents the publisher from removing its workspace.
        publish_log = dataset_root / f"{output.name}.publish.log"
        wrapped = shlex.join(["env", f"PYTHONPATH={python_path}", *map(str, command)])
        wrapped += f" > {shlex.quote(str(publish_log))} 2>&1"
        wrapped += f" && rm -- {shlex.quote(str(publish_log))}"
        job = subprocess.check_output([
            sbatch, "--parsable", f"--partition={args.partition}", f"--account={args.account}",
            "--job-name=sam31-lerobot-publish", "--nodes=1", "--ntasks=1", "--cpus-per-task=1",
            "--mem=4G", "--time=01:00:00", f"--dependency=afterok:{dependency}",
            "--kill-on-invalid-dep=yes", "--output=/dev/null", "--wrap", wrapped,
        ], text=True).strip().split(";", 1)[0]
        if not job.isdecimal():
            raise RuntimeError(f"Unexpected sbatch response: {job!r}")
        print(f"Publication job: {job} (failure log: {publish_log})", flush=True)
        print(f"Report: {report_directory / 'detection_rates.json'}", flush=True)
    else:
        print(f"Run output: {output}", flush=True)


if __name__ == "__main__":
    main()
