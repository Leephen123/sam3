#!/usr/bin/env python3
"""Repair uncertain LeRobot table/floor masks with two-sided SAM 3.1 tracking."""

import argparse
from collections import defaultdict
from contextlib import ExitStack
import fcntl
from fractions import Fraction
import gc
import json
import os
from pathlib import Path
import tempfile

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from segment_lerobot import MASK_FIELDS, MASK_NAMES, atomic_path, mask_column, mask_png, rates


SURFACES = MASK_FIELDS[:2]
MAX_DISTANCE = 32
MIN_TRACK_IOU = 0.65


def load_json(path):
    return json.loads(path.read_bytes())


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", delete=False) as temp:
        temporary = Path(temp.name)
        json.dump(data, temp)
    temporary.replace(path)


def open_frames(stack, path):
    container = stack.enter_context(av.open(str(path)))
    stream = container.streams.video[0]
    stream.thread_count = 1
    return container.decode(stream)


def read_masks(root, expected):
    labels = []
    with ExitStack() as stack:
        readers = [open_frames(stack, root / f"{name}.mkv") for name in MASK_NAMES]
        for index, frames in enumerate(zip(*readers, strict=True)):
            if any(frame.pts * frame.time_base != Fraction(index,20) for frame in frames):
                raise ValueError(f"Mask timestamps are out of sequence in {root}")
            bits = np.zeros((frames[0].height, frames[0].width), dtype=np.uint8)
            for i, frame in enumerate(frames):
                bits[frame.to_ndarray() == 255] |= 1 << i
            labels.append(bits)
    if len(labels) != expected:
        raise ValueError(f"Wrong number of frames in {root}: {len(labels)} vs {expected}")
    return np.stack(labels)


def choose_windows(labels):
    """Group suspicious frames with the same reliable anchors on each side."""
    area = labels.shape[1] * labels.shape[2]
    table = (labels & 1) != 0
    floor = (labels & 2) != 0
    overlap = np.count_nonzero(table & floor, axis=(1, 2))
    clear = (labels & 28) == 0
    n_table = np.count_nonzero(table & clear, axis=(1, 2))
    n_floor = np.count_nonzero(floor & clear & ~table, axis=(1, 2))
    minimum = np.minimum(n_table, n_floor)
    conflict = (overlap >= .05 * area) | ((overlap >= .005 * area) & (overlap > .25 * minimum))
    suspect = conflict | (minimum == 0)
    reliable = np.flatnonzero(~suspect & (minimum >= .02 * area) & (overlap <= .01 * area))
    windows = defaultdict(list)
    for i in np.flatnonzero(suspect):
        pos = np.searchsorted(reliable, i)
        if 0 < pos < len(reliable):
            left, right = map(int, reliable[pos-1:pos+1])
            if i - left <= MAX_DISTANCE and right - i <= MAX_DISTANCE:
                windows[left, right].append(int(i))
    return suspect, windows


def mask_points(mask, maximum=4):
    """Sample spaced positive points from the interior, away from boundaries."""
    interior = mask
    for _ in range(4):
        padded = np.pad(interior, 1)
        shrunk = interior & padded[2:, 1:-1] & padded[:-2, 1:-1] & padded[1:-1, 2:] & padded[1:-1, :-2]
        if not shrunk.any():
            break
        interior = shrunk
    points = np.argwhere(interior)
    h, w = mask.shape
    if not len(points):
        raise ValueError("Cannot seed an empty surface mask.")
    by_center = np.sum((points - points.mean(axis=0))**2, axis=1)
    selected = [points[int(np.argmin(by_center))]]
    min_dist = np.full(len(points), np.inf)
    for _ in range(maximum - 1):
        dist = np.sum((points - selected[-1])**2, axis=1)
        min_dist = np.minimum(min_dist, dist)
        index = int(min_dist.argmax())
        if min_dist[index] < 12**2:
            break
        selected.append(points[index])
    return [[(int(x)+.5)/w, (int(y)+.5)/h] for y, x in selected]


class SurfaceTracker:
    def __init__(self, checkpoint):
        from sam3.model_builder import build_sam3_multiplex_video_predictor

        self.predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=str(checkpoint), max_num_objects=2, use_fa3=False, use_rope_real=True,
            compile=False, warm_up=False, async_loading_frames=True,
        )
        # The local multiplex model takes a smaller set of init-state arguments
        # than the base predictor currently forwards.
        init = self.predictor.model.init_state

        def compatible_init(*, resource_path, offload_video_to_cpu, async_loading_frames, **unused):
            return init(resource_path=resource_path, offload_video_to_cpu=offload_video_to_cpu,
                        async_loading_frames=async_loading_frames)

        self.predictor.model.init_state = compatible_init

    def track(self, path, length, anchor, seed, requested, direction):
        predictor = self.predictor
        session = predictor.handle_request(dict(type="start_session", resource_path=str(path),
                                                 offload_video_to_cpu=True))["session_id"]
        results = {}
        try:
            pts = [mask_points(mask) for mask in seed]
            for index, (positive, negative) in enumerate((pts, pts[::-1]), 1):
                predictor.handle_request(dict(
                    type="add_prompt", session_id=session, frame_index=anchor, obj_id=index,
                    points=positive + negative, point_labels=[1]*len(positive) + [0]*len(negative),
                ))
            # Partial propagation merges into the cached per-frame dictionary.
            # Initialize empty dictionaries for frames without detector output.
            state = predictor._get_session(session)["state"]
            for frame in range(length):
                state["cached_frame_outputs"].setdefault(frame, {})
            for response in predictor.handle_stream_request(dict(
                type="propagate_in_video", session_id=session, start_frame_index=anchor,
                propagation_direction=direction, force_tracker_propagation=True,
            )):
                i = response["frame_index"]
                if i not in requested:
                    continue
                out = response["outputs"]
                ids = np.asarray(out["out_obj_ids"]).reshape(-1)
                masks = np.asarray(out["out_binary_masks"], dtype=bool).reshape(len(ids), *seed[0].shape)
                result = {int(j): mask for j, mask in zip(ids, masks, strict=True)}
                if 1 in result and 2 in result:
                    results[i] = np.stack([result[1], result[2]])
        finally:
            predictor.handle_request(dict(type="close_session", session_id=session, run_gc_collect=False))
        return results


def seed_at(bits):
    clear = (bits & 28) == 0
    return ((bits & 3) == 1) & clear, ((bits & 3) == 2) & clear


def agree(forward, backward, foreground):
    if forward is None or backward is None:
        return None
    intersection = forward & backward
    union = forward | backward
    support = np.count_nonzero(union, axis=(1,2))
    agreement = np.count_nonzero(intersection, axis=(1,2)) / np.maximum(support, 1)
    if min(agreement) < MIN_TRACK_IOU:
        return None
    both = intersection[0] & intersection[1]
    resolved = intersection & ~both & ~foreground
    if not resolved.any(axis=(1,2)).all():
        return None
    return resolved


def png(mask):
    return dict(bytes=mask_png(mask), path=None)


def extract_video(video, needed, folder):
    folder.mkdir()
    if not needed:
        return
    last = max(needed)
    with ExitStack() as stack:
        frames = open_frames(stack, video)
        for i, frame in enumerate(frames):
            if i in needed:
                if frame.pts * frame.time_base != Fraction(i,20):
                    raise ValueError(f"Video timestamps are out of sequence in {video}")
                frame.to_image().save(folder / f"{i:06d}.jpg", quality=96, subsampling=0)
            if i == last:
                return
    raise ValueError(f"Video ended before frame {last}: {video}")


def repair_episode(task, tracker, masks_root, original_root, repaired_root, output, scratch,
                   *, five_masks=False, packed_labels=False):
    name = task["name"]
    sidecar = Path(task["mask_parquet"])
    dest = repaired_root / sidecar.relative_to(original_root)
    report_path = output / "episodes" / f"{name}.json"
    if report_path.is_file() and dest.is_file():
        print(f"REUSE {name}", flush=True)
        return
    original = pq.read_table(sidecar, columns=["frame_index", *SURFACES])
    count = task["frames"]
    if original["frame_index"].to_pylist() != list(range(count)):
        raise ValueError(f"Invalid frame indices: {sidecar}")
    if packed_labels:
        with np.load(masks_root / name / "labels.npz", allow_pickle=False) as data:
            labels = data["labels"]
        if labels.dtype != np.uint8 or labels.ndim != 3 or labels.shape[0] != count or np.any(labels > 31):
            raise ValueError(f"Invalid packed labels: {name}")
    else:
        labels = read_masks(masks_root / name / "front_head" / "masks", count)
    suspect, windows = choose_windows(labels)
    repaired = {}
    needed = {frame for a, b in windows for frame in range(a, b+1)}
    with tempfile.TemporaryDirectory(dir=scratch, prefix=f"{name}-") as temp:
        local = Path(temp)
        frames = local / "frames"
        extract_video(task["video"], needed, frames)
        for (left, right), indices in windows.items():
            requested = {i-left for i in indices}
            with tempfile.TemporaryDirectory(dir=local, prefix="track-") as clip:
                path = Path(clip)
                for i in range(left, right+1):
                    (path / f"{i-left:05d}.jpg").symlink_to(frames / f"{i:06d}.jpg")
                length = right-left+1
                forward = tracker.track(path,length,0,seed_at(labels[left]),requested,"forward")
                backward = tracker.track(path,length,length-1,seed_at(labels[right]),requested,"backward")
                for i in indices:
                    res = agree(forward.get(i-left),backward.get(i-left),(labels[i] & 28) != 0)
                    if res is not None:
                        repaired[i] = res
    skipped = sorted(set(map(int,np.flatnonzero(suspect)))-repaired.keys())
    dest.parent.mkdir(parents=True,exist_ok=True)
    if five_masks:
        columns = {"frame_index": original["frame_index"]}
        raw_hits = {
            category: int(np.count_nonzero(((labels & (1 << index)) != 0).any(axis=(1, 2))))
            for index, category in enumerate(MASK_NAMES)
        }
        clear = (labels & 28) == 0
        table = ((labels & 1) != 0) & clear
        floor = ((labels & 2) != 0) & clear & ~table
        before_hits = dict(raw_hits)
        after_hits = dict(raw_hits)
        for index, (key, masks) in enumerate(zip(SURFACES, (table, floor), strict=True)):
            values = original[key].to_pylist()
            hit = masks.any(axis=(1, 2))
            before_hits[MASK_NAMES[index]] = int(hit.sum())
            blank = png(np.zeros(labels.shape[1:], dtype=bool))
            for i in skipped:
                values[i] = blank
                hit[i] = False
            for i, corrected in repaired.items():
                values[i] = png(corrected[index])
                hit[i] = True
            columns[key] = pa.array(values, type=original.schema.field(key).type)
            after_hits[MASK_NAMES[index]] = int(hit.sum())
        if packed_labels:
            foreground = pq.read_table(sidecar, columns=list(MASK_FIELDS[2:]))
            for key in MASK_FIELDS[2:]:
                columns[key] = foreground[key]
        else:
            for index, key in enumerate(MASK_FIELDS[2:], start=2):
                columns[key] = mask_column([mask_png((frame & (1 << index)) != 0) for frame in labels])
        with atomic_path(dest) as temp:
            pq.write_table(pa.table(columns), temp, compression="zstd", use_dictionary=False)
            metadata = pq.ParquetFile(temp)
            if metadata.metadata.num_rows != count or metadata.schema_arrow.names != ["frame_index", *MASK_FIELDS]:
                raise ValueError(f"Invalid mask Parquet: {temp}")
    elif not repaired and not skipped:
        with tempfile.TemporaryDirectory(dir=dest.parent) as directory:
            temporary = Path(directory)/dest.name
            temporary.symlink_to(sidecar)
            temporary.replace(dest)
    else:
        blank = png(np.zeros(labels.shape[1:],dtype=bool))
        columns = {"frame_index":original["frame_index"]}
        for index,key in enumerate(SURFACES):
            values = original[key].to_pylist()
            for i in skipped:
                values[i] = blank
            for i, masks in repaired.items():
                values[i] = png(masks[index])
            columns[key] = pa.array(values,type=original.schema.field(key).type)
        with tempfile.TemporaryDirectory(dir=dest.parent) as directory:
            temporary = Path(directory)/dest.name
            pq.write_table(pa.table(columns),temporary,compression="zstd",use_dictionary=False)
            if pq.ParquetFile(temporary).metadata.num_rows != count:
                raise ValueError(f"Incorrect number of rows in {temporary}")
            temporary.replace(dest)
    report = dict(ordinal=task["ordinal"], frames=count, windows=len(windows),
                  repaired=sorted(repaired), skipped=skipped, background_masks_parquet=str(dest))
    if five_masks:
        report.update(raw=rates(raw_hits, count), before=rates(before_hits, count), after=rates(after_hits, count))
    write_json(report_path, report)
    print(f"VERIFIED {name} frames={count} repaired={len(repaired)} skipped={len(skipped)} windows={len(windows)}",flush=True)


def finalize(plan, manifest, output, enriched):
    with (output/".finalize.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (output/"completed.json").is_file():
            return
        paths = [output/"episodes"/f"{task['name']}.json" for task in plan["tasks"]]
        if not all(path.is_file() for path in paths):
            return
        summary = defaultdict(int)
        data = load_json(manifest)
        if len(data["episodes"]) != len(paths):
            raise ValueError("Input manifest and repair plan differ in size.")
        for entry, task, path in zip(data["episodes"], plan["tasks"], paths, strict=True):
            report=load_json(path)
            if (report["ordinal"]!=task["ordinal"] or report["frames"]!=task["frames"]
                    or entry["background_masks_parquet"]!=task["mask_parquet"]
                    or entry["videos"]["front_head"]!=task["video"]):
                raise ValueError(f"Repair plan does not match the input manifest: {path}")
            dest=Path(report["background_masks_parquet"])
            if not dest.is_file():
                raise FileNotFoundError(dest)
            entry["background_masks_parquet"]=str(dest)
            for key in ("repaired","skipped"):
                summary[key] += len(report[key])
        write_json(enriched,data)
        write_json(output/"completed.json",dict(summary,episodes=len(paths),manifest=str(enriched)))
        print("COMPLETED",dict(summary),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("plan","manifest","masks_root","original_root","repaired_root","output","enriched","checkpoint"):
        parser.add_argument("--"+key.replace("_","-"),type=Path,required=True)
    parser.add_argument("--shard",type=int,required=True)
    parser.add_argument("--five-masks", action="store_true")
    args=parser.parse_args()
    plan=load_json(args.plan)
    tasks=[task for task in plan["tasks"] if task["shard"]==args.shard]
    if not tasks:
        raise ValueError(f"No tasks for shard {args.shard}.")
    args.output.mkdir(parents=True,exist_ok=True)
    tracker=SurfaceTracker(args.checkpoint)
    with tempfile.TemporaryDirectory(prefix="sam3-surface-repair-") as scratch:
        for task in tasks:
            repair_episode(task, tracker, args.masks_root, args.original_root, args.repaired_root,
                           args.output, scratch, five_masks=args.five_masks, packed_labels=plan.get("no_media", False))
            gc.collect()
    finalize(plan,args.manifest,args.output,args.enriched)


if __name__=="__main__":
    main()
