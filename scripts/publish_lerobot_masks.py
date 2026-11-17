#!/usr/bin/env python3
"""Validate a no-media SAM run and publish standalone masks and detection rates."""

import argparse
import fcntl
import os
from pathlib import Path
import shutil
import tempfile

from segment_lerobot import MASK_FIELDS, MASK_NAMES, load_json, mask_relative, rates, write_json


def check_report_directory(directory, dataset_root):
    directory = directory.resolve()
    if directory == dataset_root or dataset_root.is_relative_to(directory):
        raise ValueError(f"Report directory would contain source data: {directory}")
    if directory.exists() and (not directory.is_dir() or
                               any(path.name != "detection_rates.json" for path in directory.iterdir())):
        raise ValueError(f"Choose a report directory containing only detection_rates.json: {directory}")
    report = directory / "detection_rates.json"
    if report.is_file() and load_json(report).get("dataset_root") != str(dataset_root):
        raise ValueError(f"Report directory belongs to another dataset: {directory}")


def episode_metrics(task, segmentation, repair, first_pass_only):
    initial = load_json(segmentation / task["name"] / "detection_rates.json")
    if initial["frames"] != task["frames"] or initial["background_masks_parquet"] != task["mask_parquet"]:
        raise ValueError(f"Segmentation report disagrees with plan: {task['name']}")
    if first_pass_only:
        raw = initial["segmentation"]
        before = {**raw, **initial["usable_recolor_masks"]}
        after = None
        repaired = skipped = 0
    else:
        final = load_json(repair / "episodes" / f"{task['name']}.json")
        raw, before, after = (final[key] for key in ("raw", "before", "after"))
        if (final["frames"] != task["frames"] or final["ordinal"] != task["ordinal"]
                or raw != initial["segmentation"]
                or any(before[name] != initial["usable_recolor_masks"][name] for name in MASK_NAMES[:2])):
            raise ValueError(f"Repair report disagrees with segmentation: {task['name']}")
        repaired, skipped = len(final["repaired"]), len(final["skipped"])
    return dict(manifest_ordinal=task["ordinal"], episode_index=task["episode_index"],
                source_parquet=task["parquet"], frames=task["frames"],
                raw=raw, before=before, after=after, repaired=repaired, skipped=skipped)


def publish(work, dataset_root, report_directory, first_pass_only=False):
    import numpy as np
    import pyarrow.parquet as pq

    if work.parent != dataset_root or not work.name.startswith(".sam31-masks-"):
        raise ValueError(f"Expected a SAM workspace under {dataset_root}: {work}")
    check_report_directory(report_directory, dataset_root)
    submission = load_json(work / "submission.json")
    segmentation = Path(submission["segmentation"])
    plan = load_json(segmentation / "plan.json")
    tasks = plan["tasks"]
    repair = work / "repair"
    marker = load_json((segmentation if first_pass_only else repair) / "completed.json")
    manifest = load_json(submission["training_manifest"])
    selection = manifest.get("selection")
    if selection and not Path(selection).is_absolute():
        initial_manifest = Path(load_json(segmentation / "completed.json")["training_manifest"])
        manifest["selection"] = str((initial_manifest.parent / selection).resolve())
    entries = manifest["episodes"]
    if Path(marker["training_manifest" if first_pass_only else "manifest"]) != Path(submission["training_manifest"]):
        raise ValueError("Completion marker refers to a different manifest.")
    if len(tasks) != len(entries) or len(tasks) != marker["episodes"] or not tasks:
        raise ValueError("Completion marker, manifest and plan disagree about episode count.")
    output_name = "background_masks" if first_pass_only else "background_masks_tracked"
    generated_root = work / output_name
    direct_scan = plan.get("direct_scan", False)
    prepared = dataset_root / "train/prepared"
    if not (dataset_root / "train").is_dir():
        prepared = dataset_root / "prepared"
    if not direct_scan:
        prepared.mkdir(exist_ok=True)
    lock_directory = dataset_root if direct_scan else prepared

    # Keep existing manifests and mask files available to training jobs.
    run_name = work.name.removeprefix(".")
    with (lock_directory / ".sam31-publish.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        mask_root = dataset_root / output_name
        manifest_path = mask_root / "mask_index.json" if direct_scan else prepared / f"manifest_{output_name}.json"
        if direct_scan and (mask_root.is_symlink() or (mask_root.exists() and not mask_root.is_dir())):
            raise ValueError(f"Expected a local mask directory: {mask_root}")
        if mask_root.exists() or manifest_path.exists() or len(tasks) != len(load_json(plan["manifest"])["episodes"]):
            mask_root = mask_root / run_name
            manifest_path = (mask_root if direct_scan else prepared / run_name) / manifest_path.name
        reports = []
        index_entries = []
        mask_paths = {}
        train_root = dataset_root / "train"
        for task, entry in zip(tasks, entries, strict=True):
            source = Path(entry["background_masks_parquet"])
            rel = mask_relative(task, dataset_root)
            if source != generated_root / rel or source.is_symlink():
                raise ValueError(f"Expected independent run output for {task['name']}: {source}")
            if entry["videos"]["front_head"] != task["video"] or entry["episode_index"] != task["episode_index"]:
                raise ValueError(f"Output manifest refers to the wrong episode: {task['name']}")
            if pq.ParquetFile(source).schema_arrow.names != ["frame_index", *MASK_FIELDS]:
                raise ValueError(f"Unexpected mask columns: {source}")
            indices = pq.read_table(source, columns=["frame_index"])["frame_index"].to_numpy()
            if not np.array_equal(indices, np.arange(task["frames"])):
                raise ValueError(f"Mask frame indices do not match the source: {source}")
            published_rel = rel
            if not direct_scan and not first_pass_only and (dataset_root / rel).is_relative_to(train_root):
                published_rel = rel.relative_to("train")
            if published_rel in mask_paths:
                raise ValueError(f"Mask paths collide after removing train/: {published_rel}")
            mask_paths[published_rel] = rel
            entry["background_masks_parquet"] = str(mask_root / published_rel)
            if direct_scan:
                index_entries.append(dict(dataset=task["dataset"], episode_index=task["episode_index"],
                                          source_parquet=task["parquet"], front_head=task["video"],
                                          frames=task["frames"], background_masks_parquet=entry["background_masks_parquet"]))
            reports.append(episode_metrics(task, segmentation, repair, first_pass_only))

        frame_count = sum(item["frames"] for item in reports)
        summary = dict(frames=frame_count, episodes=len(reports),
                       repaired=sum(item["repaired"] for item in reports),
                       skipped=sum(item["skipped"] for item in reports))
        for stage in ("raw", "before", "after"):
            if stage == "after" and first_pass_only:
                summary[stage] = None
                continue
            hits = {name: sum(item[stage][name]["hits"] for item in reports) for name in MASK_NAMES}
            summary[stage] = rates(hits, frame_count)
        result = dict(dataset_root=str(dataset_root), camera="front_head",
                      metric="nonempty_mask_frames / total_frames",
                      manifest=str(manifest_path), summary=summary, episodes=reports)
        if direct_scan:
            manifest = dict(dataset_root=str(dataset_root), camera="front_head", total_frames=frame_count,
                            episodes=index_entries)
            result["mask_index"] = result.pop("manifest")
        report_directory.mkdir(parents=True, exist_ok=True)
        directory_fd = os.open(report_directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            fcntl.flock(directory_fd, fcntl.LOCK_EX)
            check_report_directory(report_directory, dataset_root)
            mask_root.parent.mkdir(parents=True, exist_ok=True)
            publish_root = generated_root
            if direct_scan:
                write_json(generated_root / manifest_path.name, manifest)
            if any(new != old for new, old in mask_paths.items()):
                with tempfile.TemporaryDirectory(prefix=".publish-masks-", dir=work) as scratch:
                    publish_root = Path(scratch) / output_name
                    for new, old in mask_paths.items():
                        target = publish_root / new
                        target.parent.mkdir(parents=True, exist_ok=True)
                        os.link(generated_root / old, target)
                    os.rename(publish_root, mask_root)
            else:
                os.rename(generated_root, mask_root)
            try:
                for entry in entries:
                    published = Path(entry["background_masks_parquet"])
                    if not published.is_file() or published.is_symlink():
                        raise FileNotFoundError(published)
                if not direct_scan:
                    write_json(manifest_path, manifest)
                write_json(report_directory / "detection_rates.json", result)
            except Exception:
                if not direct_scan:
                    manifest_path.unlink(missing_ok=True)
                if publish_root == generated_root:
                    os.rename(mask_root, generated_root)
                else:
                    shutil.rmtree(mask_root)
                raise
        finally:
            os.close(directory_fd)

    label = "mask index" if direct_scan else "manifest"
    print(f"Published {label}: {manifest_path}", flush=True)
    print(f"Published report: {report_directory / 'detection_rates.json'}", flush=True)
    shutil.rmtree(work)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--report-directory", type=Path, required=True)
    parser.add_argument("--segmentation-only", action="store_true")
    args = parser.parse_args()
    publish(args.work.resolve(strict=True), args.dataset_root.resolve(strict=True),
            args.report_directory.resolve(), args.segmentation_only)


if __name__ == "__main__":
    main()
