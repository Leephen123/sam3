#!/usr/bin/env python3
"""Segment front_head videos from a prepared UniFoLM LeRobot v2.1 manifest."""

import argparse
from contextlib import contextmanager
from fractions import Fraction
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import signal
import tempfile

from segment_dataset import Frame, load_detector, parse_classes, prepare_frame, run_camera, stop_worker


CAMERA = "front_head"
FPS = 20
OBJECTS = Path(__file__).with_name("fold_the_short_jeans_objects.json")
MASK_NAMES = ("table", "floor", "clothes", "pants", "robot_arm")
MASK_FIELDS = tuple(f"front_head_{name}_mask" for name in MASK_NAMES)


def mask_png(mask):
    import numpy as np
    from PIL import Image

    buffer = BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255).save(buffer, "PNG", compress_level=1)
    return buffer.getvalue()


def mask_column(pngs):
    import pyarrow as pa

    return pa.StructArray.from_arrays(
        [pa.array(pngs, type=pa.binary()), pa.nulls(len(pngs), type=pa.string())],
        names=["bytes", "path"],
    )


@contextmanager
def atomic_path(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent, prefix=f".{path.stem}.") as directory:
        temp = Path(directory) / path.name
        yield temp
        os.replace(temp, path)


def write_json(path, value):
    with atomic_path(path) as temp:
        temp.write_text(json.dumps(value, ensure_ascii=False) + "\n")


def load_json(path):
    return json.loads(Path(path).read_bytes())


def discover_lerobot(root):
    """Build an episode list from LeRobot metadata and retain its logical layout."""
    scan = root / "train" if (root / "train").is_dir() else root
    seen = set()
    episodes = []
    total_frames = 0

    def visit(path):
        nonlocal total_frames
        name = path.name.lower()
        if name.startswith(".") or "_eval" in name or name in {
            "eval", "prepared", "background_masks", "background_masks_tracked",
        }:
            return
        # Follow symlinks only when they name a LeRobot dataset, not arbitrary trees.
        info_path = path / "meta/info.json"
        if not info_path.is_file():
            if path.is_dir() and not path.is_symlink():
                for child in sorted(path.iterdir()):
                    visit(child)
            return

        dataset = path.resolve(strict=True)
        if "_eval" in dataset.name.lower() or dataset in seen:
            return
        seen.add(dataset)
        info = load_json(info_path)
        if (info["codebase_version"], info["fps"]) != ("v2.1", FPS):
            raise ValueError(f"Expected LeRobot v2.1 at {FPS} FPS: {path}")
        rows = sorted(
            (json.loads(line) for line in (dataset / "meta/episodes.jsonl").read_text().splitlines()
             if line.strip()),
            key=lambda row: row["episode_index"],
        )
        if (not rows or len({row["episode_index"] for row in rows}) != len(rows)
                or len(rows) != info["total_episodes"]
                or sum(row["length"] for row in rows) != info["total_frames"]):
            raise ValueError(f"Inconsistent LeRobot episode metadata: {path}")
        for row in rows:
            index, count = row["episode_index"], row["length"]
            if type(index) is not int or index < 0 or type(count) is not int or count < 1:
                raise ValueError(f"Invalid episode index or length: {path}")
            fields = dict(episode_chunk=index // info["chunks_size"], episode_index=index)
            data_rel = Path(info["data_path"].format(**fields))
            video_rel = Path(info["video_path"].format(**fields, video_key=CAMERA))
            if any(p.is_absolute() or ".." in p.parts or not p.parts for p in (data_rel, video_rel)):
                raise ValueError(f"Expected paths relative to the LeRobot dataset: {path}")
            video = (dataset / video_rel).resolve(strict=True)
            episodes.append(dict(dataset=str(dataset), episode_index=index,
                                 videos={CAMERA: str(video)},
                                 mask_relative=str(path.relative_to(scan) / data_rel)))
            total_frames += count

    visit(scan)
    if not episodes:
        raise ValueError(f"No LeRobot training datasets found under {scan}")
    return dict(episodes=episodes, total_frames=total_frames)


def mask_relative(task, root):
    if "mask_relative" not in task:
        return Path(task["parquet"]).relative_to(root)
    relative = Path(task["mask_relative"])
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"Invalid relative mask path: {relative}")
    return relative


def prepare(args):
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    root = args.dataset_root.resolve() if args.dataset_root else args.manifest.resolve().parent.parent
    entries = manifest["episodes"]
    ordinals = list(range(len(entries))) if args.ordinals is None else args.ordinals
    if len(set(ordinals)) != len(ordinals) or any(not 0 <= i < len(entries) for i in ordinals):
        raise ValueError("Choose distinct in-range manifest ordinals.")
    if not 1 <= args.shards <= len(ordinals):
        raise ValueError("Shards must be between one and the number of selected episodes.")
    objects = json.loads(OBJECTS.read_text())
    parse_classes(json.dumps(objects))

    tasks = []
    metadata = {}
    mask_paths = set()
    seen = set()
    for ordinal in sorted(ordinals):
        entry = entries[ordinal]
        dataset = Path(entry["dataset"]).resolve()
        if dataset not in metadata:
            info = load_json(dataset / "meta" / "info.json")
            if info["codebase_version"] != "v2.1" or info["fps"] != FPS:
                raise ValueError(f"Expected LeRobot v2.1 at {FPS} FPS: {dataset}")
            episode_rows = (dataset / "meta" / "episodes.jsonl").read_text().splitlines()
            lengths = {episode["episode_index"]: episode["length"] for episode in map(json.loads, episode_rows)}
            metadata[dataset] = info, lengths
        info, lengths = metadata[dataset]
        index = entry["episode_index"]
        count = lengths[index]
        parquet = dataset / info["data_path"].format(
            episode_chunk=index // info["chunks_size"], episode_index=index,
        )
        video = Path(entry["videos"][CAMERA])
        if parquet in seen or not parquet.is_file() or not video.is_file() or count < 1:
            raise ValueError(f"Duplicate or missing source: {parquet}, {video}")
        seen.add(parquet)
        task = dict(ordinal=ordinal, name=f"episode_{ordinal:06d}", episode_index=index,
                    frames=count, parquet=str(parquet), video=str(video))
        if args.direct_scan:
            task.update(dataset=str(dataset), mask_relative=entry["mask_relative"], allow_timestamp_gaps=True)
        sidecar = args.mask_root.resolve() / mask_relative(task, root)
        if sidecar in mask_paths:
            raise ValueError(f"Conflicting episode mask path: {sidecar}")
        mask_paths.add(sidecar)
        task["mask_parquet"] = str(sidecar)
        tasks.append(task)

    frame_total = sum(task["frames"] for task in tasks)
    if args.ordinals is None and frame_total != manifest["total_frames"]:
        raise ValueError(f"Episode lengths total {frame_total}, manifest reports {manifest['total_frames']}.")
    loads = [0] * args.shards
    for task in sorted(tasks, key=lambda item: item["frames"], reverse=True):
        shard = loads.index(min(loads))
        task["shard"] = shard
        loads[shard] += task["frames"]
    plan = dict(manifest=str(args.manifest.resolve()), manifest_sha256=sha256(manifest_bytes).hexdigest(),
                compile=False, fa3=False, batch_size=4, camera=CAMERA, fps=FPS, objects=objects,
                frames=frame_total, loads=loads, tasks=tasks)
    if args.no_media:
        plan["no_media"] = True
        plan["dataset_root"] = str(root)
    if args.direct_scan:
        plan["direct_scan"] = True
    if args.enriched_manifest is not None:
        enriched = args.enriched_manifest.resolve()
        if enriched == args.manifest.resolve():
            raise ValueError("The enriched manifest must have a different path from the input manifest.")
        plan["enriched_manifest"] = str(enriched)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "plan.json"
    if path.exists() and load_json(path) != plan:
        raise ValueError(f"Output already contains a different plan: {path}")
    if not path.exists():
        write_json(path, plan)
    print(f"Prepared {len(tasks)} episodes, {frame_total} frames, shard frame counts {loads}.", flush=True)


def validate_parquet(task):
    import numpy as np
    import pyarrow.parquet as pq

    table = pq.read_table(task["parquet"], columns=["episode_index", "frame_index", "timestamp"])
    count = task["frames"]
    if (len(table) != count or any(column.null_count for column in table.columns)
            or not np.array_equal(table["frame_index"].to_numpy(), np.arange(count))
            or not np.all(table["episode_index"].to_numpy() == task["episode_index"])):
        raise ValueError(f"Unexpected frame indices: {task['parquet']}")
    timestamps = table["timestamp"].to_numpy().astype(np.float64)
    ticks = np.rint(timestamps * FPS)
    expected = np.arange(count)
    if task.get("allow_timestamp_gaps"):
        if not np.isfinite(ticks).all() or ticks[0] < 0 or np.any(np.diff(ticks) <= 0):
            raise ValueError(f"Unexpected frame timestamps: {task['parquet']}")
        expected = ticks
    if not np.allclose(timestamps, expected / FPS, atol=1e-4, rtol=0):
        raise ValueError(f"Unexpected frame timestamps: {task['parquet']}")


class VideoBatches:
    """Keep one decoded batch available if inference retries with a smaller batch."""

    def __init__(self, container, path):
        self.path = path
        stream = container.streams.video[0]
        if stream.average_rate != FPS:
            raise ValueError(f"Unexpected frame rate {stream.average_rate}: {path}")
        stream.thread_count = 2
        self.time_base = stream.time_base
        self.frames = container.decode(stream)
        self.pending = []
        self.start = 0
        self.next_index = 0

    def read(self, start, count):
        offset = start - self.start
        if not 0 <= offset <= len(self.pending):
            raise RuntimeError(f"Unexpected decoder offset {start}: {self.path}")
        del self.pending[:offset]
        self.start = start
        while len(self.pending) < count:
            frame = next(self.frames, None)
            index = self.next_index
            if frame is None or frame.pts is None or frame.pts * self.time_base != Fraction(index, FPS):
                raise ValueError(f"Unexpected or missing video frame {index}: {self.path}")
            self.pending.append(prepare_frame(frame.to_image()))
            self.next_index += 1
        return self.pending[:count]

    def verify_end(self, count):
        if self.next_index != count or next(self.frames, None) is not None:
            raise ValueError(f"Video frame count differs from its Parquet: {self.path}")


class BackgroundMasks:
    def __init__(self, classes, *, keep_labels=False):
        self.flags = {item["name"]: item["flag"] for item in classes}
        self.foreground = self.flags["clothes"] | self.flags["pants"] | self.flags["robot_arm"]
        names = MASK_NAMES if keep_labels else MASK_NAMES[:2]
        self.pngs = {name: [] for name in names}
        self.hits = dict.fromkeys(MASK_NAMES[:2], 0)
        self.labels = [] if keep_labels else None

    def add(self, start, labels):
        import numpy as np

        if start != len(self.pngs["table"]):
            raise ValueError(f"Nonsequential mask batch at frame {start}.")
        clear = (labels & self.foreground) == 0
        table = ((labels & self.flags["table"]) != 0) & clear
        floor = ((labels & self.flags["floor"]) != 0) & clear & ~table
        surfaces = {"table": table, "floor": floor}
        for name, masks in surfaces.items():
            self.hits[name] += int(np.count_nonzero(masks.any(axis=(1, 2))))
        for name, pngs in self.pngs.items():
            masks = surfaces[name] if name in surfaces else (labels & self.flags[name]) != 0
            pngs.extend(mask_png(mask) for mask in masks)
        if self.labels is not None:
            # Tracking uses a fixed bit order independent of prompt configuration order.
            packed = np.zeros_like(labels)
            for index, name in enumerate(MASK_NAMES):
                packed[(labels & self.flags[name]) != 0] |= 1 << index
            self.labels.append(packed)

    def save(self, path, count):
        import pyarrow as pa
        import pyarrow.parquet as pq

        fields = {"frame_index": pa.array(range(count), type=pa.int64())}
        for name, masks in self.pngs.items():
            if len(masks) != count:
                raise ValueError(f"Incorrect number of {name} masks: {len(masks)}.")
            fields[f"front_head_{name}_mask"] = mask_column(masks)
        with atomic_path(path) as temp:
            pq.write_table(pa.table(fields), temp, compression="zstd", use_dictionary=False)
            if pq.ParquetFile(temp).metadata.num_rows != count:
                raise ValueError(f"Failed to verify the saved masks: {path}")


def rates(hits, count):
    return {name: dict(hits=value, rate=value / count) for name, value in hits.items()}


def segment(task, output, model, classes, prompts, text, batch_size, *, no_media=False):
    import av
    import numpy as np

    name = task["name"]
    count = task["frames"]
    target = output / name
    sidecar = Path(task["mask_parquet"])
    if target.exists():
        report = load_json(target / "detection_rates.json")
        products = [sidecar]
        if no_media:
            products.append(target / "labels.npz")
        else:
            products.extend((target / CAMERA / "overlay.mp4", target / CAMERA / "preview.jpg"))
            products.extend(target / CAMERA / "masks" / f"{item['name']}.mkv" for item in classes)
        if report["frames"] != count or report["background_masks_parquet"] != str(sidecar) or not all(
            path.is_file() and path.stat().st_size > 0 for path in products
        ):
            raise ValueError(f"Inconsistent previously published episode: {target}")
        print(f"REUSE {name} {count}", flush=True)
        return
    validate_parquet(task)
    frames = [Frame(name, CAMERA, Path(f"{index:06d}")) for index in range(count)]
    masks = BackgroundMasks(classes, keep_labels=no_media)
    with tempfile.TemporaryDirectory(prefix=f".{name}.", dir=output / ".partial") as directory:
        stage = Path(directory)
        with av.open(task["video"]) as container:
            reader = VideoBatches(container, task["video"])
            hits, _ = run_camera(
                0, model, frames, classes, prompts, text, batch_size, FPS, stage, None,
                read_batch=reader.read, observe_masks=masks.add, write_media=not no_media,
            )
            reader.verify_end(count)
        masks.save(sidecar, count)
        if no_media:
            episode_dir = stage / name
            episode_dir.mkdir()
            np.savez_compressed(episode_dir / "labels.npz", labels=np.concatenate(masks.labels))
        write_json(stage / name / "detection_rates.json", dict(
            manifest_ordinal=task["ordinal"], episode_index=task["episode_index"], frames=count,
            source_parquet=task["parquet"], camera=CAMERA, background_masks_parquet=str(sidecar),
            segmentation=rates({item["name"]: hit for item, hit in zip(classes, hits)}, count),
            usable_recolor_masks=rates(masks.hits, count),
        ))
        os.rename(stage / name, target)
    print(f"VERIFIED {name} {count}", flush=True)


def finalize(output, plan):
    import fcntl

    tasks = plan["tasks"]
    with (output / ".finalize.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (output / "completed.json").exists():
            return
        report_paths = [output / task["name"] / "detection_rates.json" for task in tasks]
        if not all(path.is_file() for path in report_paths):
            return
        manifest_path = Path(plan["manifest"])
        manifest_bytes = manifest_path.read_bytes()
        if sha256(manifest_bytes).hexdigest() != plan["manifest_sha256"]:
            raise ValueError(f"Source manifest changed during the run: {manifest_path}")
        manifest = json.loads(manifest_bytes)
        episodes = []
        reports = []
        for task, path in zip(tasks, report_paths):
            report = load_json(path)
            if (report["frames"] != task["frames"] or report["background_masks_parquet"] != task["mask_parquet"]
                    or not Path(task["mask_parquet"]).is_file()):
                raise ValueError(f"Inconsistent episode report: {path}")
            reports.append(report)
            entry = manifest["episodes"][task["ordinal"]]
            entry["background_masks_parquet"] = task["mask_parquet"]
            episodes.append(entry)
        manifest["episodes"] = episodes
        manifest["total_frames"] = plan["frames"]
        if not plan.get("direct_scan"):
            manifest["valid_anchors"] = sum(len(entry["valid_rows"]) for entry in episodes)
        if "enriched_manifest" in plan:
            enriched = Path(plan["enriched_manifest"])
        elif len(episodes) != len(json.loads(manifest_bytes)["episodes"]):
            enriched = output / "manifest_background_masks.json"
        else:
            enriched = manifest_path.with_name("manifest_background_masks.json")
        selection = manifest.get("selection")
        if selection and enriched.parent != manifest_path.parent:
            manifest["selection"] = str((manifest_path.parent / selection).resolve())
        write_json(output / "episode_detection_rates.json", dict(
            metric="nonempty_mask_frames / total_frames", camera=CAMERA,
            frames=plan["frames"], episodes=reports,
        ))
        write_json(enriched, manifest)
        write_json(output / "completed.json", dict(
            episodes=len(episodes), frames=plan["frames"], training_manifest=str(enriched),
        ))
        print(f"COMPLETED {len(episodes)} episodes and {plan['frames']} frames.", flush=True)


def worker(args):
    import torch

    output = args.output
    plan = load_json(output / "plan.json")
    if not 0 <= args.shard < len(plan["loads"]):
        raise ValueError(f"Invalid shard {args.shard}.")
    if plan["compile"] or plan["fa3"]:
        raise ValueError("The LeRobot worker uses eager mode with PyTorch SDPA.")
    if not plan.get("no_media") and not all(shutil.which(tool) for tool in ("ffmpeg", "ffprobe")):
        raise ValueError("FFmpeg and ffprobe are required.")
    if torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one GPU to each worker.")
    (output / ".partial").mkdir(exist_ok=True)
    signal.signal(signal.SIGTERM, stop_worker)
    classes = parse_classes(json.dumps(plan["objects"]))
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    model = load_detector(args.checkpoint, compile_model=False, use_fa3=False)
    prompts = [(i, prompt["text"], prompt["threshold"])
               for i, item in enumerate(classes) for prompt in item["prompts"]]
    tasks = [task for task in plan["tasks"] if task["shard"] == args.shard]
    print(f"READY shard={args.shard} episodes={len(tasks)} frames={plan['loads'][args.shard]} "
          "torch_compile=off fa3=off", flush=True)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        text = model.backbone.forward_text([prompt[1] for prompt in prompts], device="cuda")
        for task in tasks:
            print(f"START {task['name']} {task['parquet']}", flush=True)
            segment(task, output, model, classes, prompts, text, plan["batch_size"],
                    no_media=plan.get("no_media", False))
    finalize(output, plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--manifest", required=True, type=Path)
    prepare_parser.add_argument("--output", required=True, type=Path)
    prepare_parser.add_argument("--mask-root", required=True, type=Path)
    prepare_parser.add_argument("--shards", required=True, type=int)
    prepare_parser.add_argument("--enriched-manifest", type=Path,
                                help="Write the mask-enriched manifest to this path.")
    prepare_parser.add_argument("--ordinals", nargs="+", type=int)
    prepare_parser.add_argument("--no-media", action="store_true")
    prepare_parser.add_argument("--direct-scan", action="store_true", help=argparse.SUPPRESS)
    prepare_parser.add_argument("--dataset-root", type=Path)
    worker_parser = sub.add_parser("worker")
    worker_parser.add_argument("--output", required=True, type=Path)
    worker_parser.add_argument("--checkpoint", required=True, type=Path)
    worker_parser.add_argument("--shard", required=True, type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        worker(args)


if __name__ == "__main__":
    main()
