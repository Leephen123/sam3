"""Write SAM 3.1 segmentation videos for a G1 episode dataset."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass
from fractions import Fraction
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty
import shutil
import signal
import subprocess
import tempfile


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
RESOLUTION = 1008
COLORS = (
    (55, 139, 246), (244, 79, 164), (252, 170, 24), (76, 204, 117),
    (24, 221, 232), (165, 105, 235), (244, 90, 74), (201, 212, 64),
    (88, 122, 187), (235, 143, 126), (128, 194, 147), (232, 189, 220),
    (176, 139, 102), (162, 194, 245), (210, 195, 164), (178, 182, 192),
)


@dataclass(frozen=True)
class Frame:
    episode: str
    camera: str
    source: Path


def parse_classes(value):
    try:
        items = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"Invalid objects JSON: {error}") from error
    if not isinstance(items, list) or not 1 <= len(items) <= len(COLORS):
        raise argparse.ArgumentTypeError("Specify 1–16 objects in a JSON list.")

    classes = []
    names = set()
    for index, item in enumerate(items):
        name = item.get("name") if isinstance(item, dict) else None
        prompts = item.get("prompts") if isinstance(item, dict) else None
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            raise argparse.ArgumentTypeError("Each object needs a nonempty name without surrounding spaces.")
        if name in {".", ".."} or any(separator in name for separator in ("/", chr(92), chr(0))):
            raise argparse.ArgumentTypeError(f"Invalid object filename: {name!r}")
        if name in names:
            raise argparse.ArgumentTypeError(f"Duplicate object name: {name}")
        if not isinstance(prompts, dict) or not prompts:
            raise argparse.ArgumentTypeError(f"{name} needs a nonempty prompts object.")
        normalized = []
        for text, score in prompts.items():
            if (
                not text.strip()
                or type(score) not in (float, int)
                or not 0 <= score <= 1
                or not math.isfinite(score)
            ):
                raise argparse.ArgumentTypeError(f"Invalid prompt or confidence for {name}: {text!r} {score!r}")
            normalized.append({"text": text.strip(), "threshold": float(score)})
        names.add(name)
        classes.append({"name": name, "flag": 1 << index, "color": COLORS[index], "prompts": normalized})
    return classes


def list_frames(root, requested_episodes, cameras):
    if not root.is_dir():
        raise ValueError(f"Dataset path does not exist: {root}")
    episodes = {path.name: path for path in root.glob("episode_*") if path.is_dir()}
    if not episodes:
        raise ValueError(f"No episode_* directories under {root}")
    requested = requested_episodes or sorted(episodes)
    if len(set(requested)) != len(requested) or len(set(cameras)) != len(cameras):
        raise ValueError("Episode and camera names must be unique.")
    frames = []
    for episode in requested:
        if episode not in episodes:
            raise ValueError(f"Episode does not exist in dataset: {episode}")
        for camera in cameras:
            if Path(camera).name != camera or camera in {".", ".."}:
                raise ValueError(f"Invalid camera name: {camera}")
            folder = episodes[episode] / "images" / camera
            if not folder.is_dir():
                raise ValueError(f"Camera directory does not exist: {folder}")
            with os.scandir(folder) as entries:
                images = sorted(
                    folder / entry.name for entry in entries
                    if os.path.splitext(entry.name)[1].lower() in IMAGE_EXTENSIONS and entry.is_file()
                )
            if not images:
                raise ValueError(f"No images in {folder}")
            frames.extend(Frame(episode, camera, image) for image in images)
    return frames


def load_detector(checkpoint, *, compile_model=False, use_fa3=False):
    import torch

    from sam3 import model_builder as builder
    from sam3.model.sam3_multiplex_detector import Sam3MultiplexDetector
    from sam3.model.vl_combiner import SAM3VLBackboneTri

    # Use the detector that the 3.1 video predictor builds. Independent frame
    # segmentation needs the detector weights and can skip the video tracker.
    vocab = Path(builder.__file__).parent / "assets/bpe_simple_vocab_16e6.txt.gz"
    if use_fa3:
        try:
            from flash_attn_interface import flash_attn_func
        except ImportError as error:
            raise RuntimeError("--fa3 requires a compatible FlashAttention 3 installation.") from error
        if torch.cuda.get_device_capability()[0] != 9:
            raise RuntimeError("--fa3 requires an NVIDIA Hopper GPU.")
    vision = builder._create_multiplex_tri_backbone(
        compile_mode="default" if compile_model else None,
        use_fa3=use_fa3, use_rope_real=True,
    )
    detector = Sam3MultiplexDetector(
        num_feature_levels=1,
        backbone=SAM3VLBackboneTri(
            scalp=0, visual=vision, text=builder._create_text_encoder(str(vocab))
        ),
        transformer=builder._create_sam3_transformer(use_fa3=use_fa3),
        segmentation_head=builder._create_segmentation_head(use_fa3=use_fa3),
        semantic_segmentation_head=None,
        input_geometry_encoder=builder._create_geometry_encoder(),
        use_early_fusion=True,
        use_dot_prod_scoring=True,
        dot_prod_scoring=builder._create_dot_product_scoring(),
        supervise_joint_box_scores=True,
        is_multiplex=True,
    )
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    checkpoint_data = checkpoint_data.get("model", checkpoint_data)
    weights = {
        name.split(".", 1)[1]: value
        for name, value in checkpoint_data.items()
        if name.startswith(("detector.", "sam3_model."))
    }
    missing, unexpected = detector.load_state_dict(weights, strict=False)
    missing = [name for name in missing if not name.endswith(("freqs_cis_real", "freqs_cis_imag"))]
    if missing or unexpected:
        raise ValueError(
            f"Incompatible SAM 3.1 checkpoint. Missing detector keys ({len(missing)}): "
            f"{missing[:5]}. Unexpected keys ({len(unexpected)}): {unexpected[:5]}."
        )
    return detector.cuda().eval()


def read_frame(frame):
    from PIL import Image

    with Image.open(frame.source) as image:
        return prepare_frame(image)


def prepare_frame(image):
    import numpy as np
    from PIL import Image

    original = image.convert("RGB")
    resized = original.resize((RESOLUTION, RESOLUTION), Image.Resampling.BILINEAR)
    return np.asarray(original), np.asarray(resized)


def union_masks(logits, keep, size):
    import torch
    from torch.nn import functional as F

    width, height = size
    counts = torch.zeros((len(logits), height, width), dtype=torch.int32, device=logits.device)
    selected = keep.nonzero()
    if len(selected):
        image_ids, query_ids = selected.unbind(1)
        resized = F.interpolate(
            logits[image_ids, query_ids, None].float(),
            size=(height, width), mode="bilinear", align_corners=False,
        )
        counts.index_add_(0, image_ids, (resized[:, 0] > 0).to(torch.int32))
    return counts != 0


def segment_batch(detector, images, size, text_features, prompts, class_count):
    import numpy as np
    import torch

    from sam3.model.data_misc import FindStage

    frame_count = len(images)
    batch = torch.from_numpy(np.stack(images)).cuda()
    batch = batch.permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)
    features = detector.backbone.forward_image(
        batch, need_interactive_out=False, need_propagation_out=False
    )
    features.update(text_features)

    # Each prompt queries the same batch of frames, in prompt-major order.
    find_input = FindStage(
        img_ids=torch.arange(frame_count, device="cuda").repeat(len(prompts)),
        text_ids=torch.arange(len(prompts), device="cuda").repeat_interleave(frame_count),
        input_boxes=None,
        input_boxes_mask=None,
        input_boxes_label=None,
        input_points=None,
        input_points_mask=None,
    )
    result = detector.forward_grounding(
        backbone_out=features,
        find_input=find_input,
        find_target=None,
        geometric_prompt=detector._get_dummy_prompt(len(find_input.img_ids)),
    )
    scores = result["pred_logits"].squeeze(-1).float().sigmoid()
    logits = result["pred_masks"]
    width, height = size
    labels = torch.zeros((frame_count, height, width), dtype=torch.int32, device="cuda")
    for query_index, (class_index, _, threshold) in enumerate(prompts):
        rows = slice(query_index * frame_count, (query_index + 1) * frame_count)
        masks = union_masks(logits[rows], scores[rows] >= threshold, size)
        labels.bitwise_or_(masks.to(torch.int32).mul_(1 << class_index))
    dtype = np.uint8 if class_count <= 8 else np.uint16
    return labels.cpu().numpy().astype(dtype)


@contextmanager
def video_encoder(path, size, fps, *, mask):
    width, height = size
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
        "-filter_threads", "1", "-f", "rawvideo", "-pixel_format", "gray" if mask else "rgb24",
        "-video_size", f"{width}x{height}", "-framerate", str(fps), "-i", "pipe:0",
        "-an", "-threads", "1",
    ]
    if mask:
        command += ["-c:v", "ffv1", "-level", "3", "-g", "1", "-slicecrc", "1", "-pix_fmt", "gray"]
    else:
        if width % 2 or height % 2:
            command += ["-vf", f"pad={width + width % 2}:{height + height % 2}"]
        command += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart"]

    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command + [str(path)], stdin=subprocess.PIPE, stderr=errors)
        try:
            yield process.stdin
            process.stdin.close()
            if process.wait(timeout=60):
                errors.seek(0)
                raise RuntimeError(f"FFmpeg failed for {path}: {errors.read().decode(errors='replace').strip()}")
        except BaseException as error:
            if process.poll() is None:
                process.kill()
            process.wait()
            if isinstance(error, (BrokenPipeError, subprocess.TimeoutExpired)):
                errors.seek(0)
                detail = errors.read().decode(errors="replace").strip()
                raise RuntimeError(f"FFmpeg stopped writing {path}: {detail or error}") from error
            raise
        finally:
            if not process.stdin.closed:
                with suppress(BrokenPipeError):
                    process.stdin.close()


def verify_video(path, size, fps, frame_count, codec, pixel_format):
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=codec_name,width,height,pix_fmt,avg_frame_rate,nb_read_packets",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True, timeout=60,
    )
    streams = json.loads(probe.stdout)["streams"]
    if len(streams) != 1:
        raise ValueError(f"Expected one video stream in {path}")
    stream = streams[0]
    actual = (
        stream["codec_name"], (stream["width"], stream["height"]), stream["pix_fmt"],
        Fraction(stream["avg_frame_rate"]), int(stream["nb_read_packets"]),
    )
    expected = (codec, size, pixel_format, Fraction(str(fps)), frame_count)
    if actual != expected:
        raise ValueError(f"Incorrect video metadata for {path}: {actual}; expected {expected}")


def write_video_batch(overlay_stream, mask_streams, classes, frames, originals, labels,
                      start, preview_indexes, samples, hits, observe_masks=None):
    import numpy as np

    for index, (frame, original, flags) in enumerate(zip(frames, originals, labels), start):
        pixel_colors = np.zeros(original.shape, dtype=np.uint16)
        color_counts = np.zeros(flags.shape, dtype=np.uint8)
        for class_index, (item, stream) in enumerate(zip(classes, mask_streams)):
            active = (flags & item["flag"]) != 0
            stream.write((active.astype(np.uint8) * 255).tobytes())
            hits[class_index] += int(active.any())
            pixel_colors[active] += np.asarray(item["color"], dtype=np.uint16)
            color_counts[active] += 1

        overlay = original.copy()
        active = color_counts != 0
        overlay[active] = np.rint(
            original[active] * 0.48 + (pixel_colors[active] / color_counts[active, None]) * 0.52
        ).astype(np.uint8)
        overlay_stream.write(overlay.tobytes())
        if index in preview_indexes:
            samples.append((frame.source.name, original, overlay))
    if observe_masks is not None:
        observe_masks(start, labels)


def render_preview(samples, target, classes):
    from PIL import Image, ImageDraw

    tile_width, tile_height = 310, 176
    legend_rows = (len(classes) + 3) // 4
    image_top = legend_rows * 20 + 23
    sheet = Image.new(
        "RGB", (max(620, len(samples) * tile_width), image_top + 2 * tile_height + 12), "#f5f5f5"
    )
    draw = ImageDraw.Draw(sheet)
    for index, item in enumerate(classes):
        x, y = 9 + (index % 4) * 151, 8 + (index // 4) * 20
        draw.rectangle((x, y, x + 11, y + 11), fill=item["color"])
        draw.text((x + 16, y - 2), item["name"], fill="#202020")
    for col, (name, original, overlay) in enumerate(samples):
        x = col * tile_width
        draw.text((x + 7, image_top - 17), name, fill="#202020")
        for pixels, y in ((original, image_top), (overlay, image_top + tile_height + 5)):
            image = Image.fromarray(pixels)
            image.thumbnail((tile_width, tile_height))
            sheet.paste(image, (x, y))
    sheet.save(target / "preview.jpg", quality=87)


def run_camera(gpu, model, frames, classes, prompts, text, batch_size, fps, stage, readers,
               *, read_batch=None, observe_masks=None, write_media=True):
    import numpy as np
    import torch

    first_frame = frames[0]
    target = stage / first_frame.episode / first_frame.camera
    masks_dir = target / "masks"
    if write_media:
        masks_dir.mkdir(parents=True)
    mask_paths = [masks_dir / f"{item['name']}.mkv" for item in classes]
    overlay_path = target / "overlay.mp4"
    preview_indexes = set(np.linspace(0, len(frames) - 1, min(4, len(frames)), dtype=int))
    samples = []
    hits = [0] * len(classes)
    next_progress = 240

    with ExitStack() as stack:
        previous_write = None
        position = 0
        while position < len(frames):
            block = frames[position:position + batch_size]
            prepared = read_batch(position, len(block)) if read_batch else readers.map(read_frame, block)
            originals, images = zip(*prepared)
            if position == 0:
                size = originals[0].shape[1::-1]
                if write_media:
                    overlay_stream = stack.enter_context(video_encoder(overlay_path, size, fps, mask=False))
                    mask_streams = [
                        stack.enter_context(video_encoder(path, size, fps, mask=True)) for path in mask_paths
                    ]
                    writer = stack.enter_context(ThreadPoolExecutor(max_workers=1))
            if any(image.shape[1::-1] != size for image in originals):
                raise ValueError(f"Frame dimensions changed in {first_frame.episode}/{first_frame.camera}")

            while True:
                try:
                    labels = segment_batch(model, images, size, text, prompts, len(classes))
                except torch.cuda.OutOfMemoryError:
                    if len(block) == 1:
                        raise
                else:
                    break
                batch_size = max(1, len(block) // 2)
                block, originals, images = block[:batch_size], originals[:batch_size], images[:batch_size]
                torch.cuda.empty_cache()
                print(f"[GPU {gpu}] CUDA memory limit reached, using batch size {batch_size}.", flush=True)

            if write_media:
                # At most one batch is being encoded while the next is in inference.
                if previous_write is not None:
                    previous_write.result()
                previous_write = writer.submit(
                    write_video_batch, overlay_stream, mask_streams, classes, block, originals, labels,
                    position, preview_indexes, samples, hits, observe_masks,
                )
            else:
                for index, item in enumerate(classes):
                    hits[index] += int(np.count_nonzero(((labels & item["flag"]) != 0).any(axis=(1, 2))))
                if observe_masks is not None:
                    observe_masks(position, labels)
            position += len(block)
            if position >= next_progress:
                print(f"[GPU {gpu}] {first_frame.episode}/{first_frame.camera} {position}/{len(frames)}",
                      flush=True)
                next_progress = position + 240
        if previous_write is not None:
            previous_write.result()

    if not write_media:
        return hits, batch_size

    overlay_size = tuple(length + length % 2 for length in size)
    verify_video(overlay_path, overlay_size, fps, len(frames), "h264", "yuv420p")
    for path in mask_paths:
        verify_video(path, size, fps, len(frames), "ffv1", "gray")
    render_preview(samples, target, classes)
    return hits, batch_size


def stop_worker(signum, _frame):
    raise SystemExit(128 + signum)


def run_worker(gpu, camera_groups, classes, checkpoint, batch_size, fps, stage, reports,
               compile_model=False, use_fa3=False, compile_cache=None):
    signal.signal(signal.SIGTERM, stop_worker)
    if compile_cache is not None:
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(compile_cache / "inductor")
        os.environ["TRITON_CACHE_DIR"] = str(compile_cache / "triton")

    import torch

    torch.set_num_threads(2)
    if gpu >= torch.cuda.device_count():
        raise ValueError(f"GPU {gpu} is unavailable. Visible devices: {torch.cuda.device_count()}.")
    torch.cuda.set_device(gpu)
    model = load_detector(checkpoint, compile_model=compile_model, use_fa3=use_fa3)
    prompts = [
        (class_index, prompt["text"], prompt["threshold"])
        for class_index, item in enumerate(classes)
        for prompt in item["prompts"]
    ]
    frame_count = sum(map(len, camera_groups))
    print(f"[GPU {gpu}] Ready. {frame_count} frames across {len(camera_groups)} cameras.", flush=True)

    with (
        ThreadPoolExecutor(max_workers=4) as readers,
        torch.inference_mode(),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
    ):
        text = model.backbone.forward_text([prompt[1] for prompt in prompts], device="cuda")
        for frames in camera_groups:
            hits, batch_size = run_camera(
                gpu, model, frames, classes, prompts, text, batch_size, fps, stage, readers
            )
            reports.put((frames[0].episode, frames[0].camera, len(frames), hits))
    peak = torch.cuda.max_memory_reserved() / 1024 ** 3
    print(f"[GPU {gpu}] Peak reserved GPU memory: {peak:.2f} GiB.", flush=True)
    if compile_model:
        from torch._dynamo.utils import counters

        graphs = counters["stats"]["unique_graphs"]
        if not graphs:
            raise RuntimeError("torch.compile was requested but no graph was captured.")
        print(f"[GPU {gpu}] torch.compile graphs: {graphs}", flush=True)


def assign_groups(groups, gpus):
    assignments = {gpu: [] for gpu in gpus}
    loads = dict.fromkeys(gpus, 0)
    for frames in sorted(groups.values(), key=len, reverse=True):
        gpu = min(loads, key=loads.get)
        assignments[gpu].append(frames)
        loads[gpu] += len(frames)
    return assignments


def collect_reports(processes, reports, groups):
    results = {}
    while len(results) < len(groups):
        try:
            episode, camera, count, hits = reports.get(timeout=0.5)
        except Empty:
            failed = next((p for p in processes if p.exitcode not in (None, 0)), None)
            if failed is not None:
                raise RuntimeError(f"{failed.name} failed (exit {failed.exitcode}).")
            if all(p.exitcode is not None for p in processes):
                raise RuntimeError(f"Only received {len(results)} of {len(groups)} camera reports.")
            continue
        key = episode, camera
        if key not in groups or key in results or count != len(groups[key]):
            raise ValueError(f"Incorrect camera report for {episode}/{camera}")
        results[key] = (count, hits)
        print(f"Verified {episode}/{camera}: {count} frames.", flush=True)

    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"{process.name} failed (exit {process.exitcode}).")
    return results


def print_reports(results, classes):
    episodes = defaultdict(Counter)
    frame_counts = Counter()
    names = [item["name"] for item in classes]

    def print_row(label, total, hits):
        rates = " ".join(f"{name}={count}/{total}({count / total:.2%})" for name, count in zip(names, hits))
        print(f"{label} {rates}", flush=True)

    for (episode, camera), (total, hits) in sorted(results.items()):
        frame_counts[episode] += total
        episodes[episode].update(dict(zip(names, hits)))
        print_row(f"CAMERA {episode} {camera}", total, hits)
    for episode, counts in episodes.items():
        print_row(f"EPISODE {episode}", frame_counts[episode], [counts[name] for name in names])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="G1 dataset root containing episode_* directories")
    parser.add_argument(
        "--output", type=Path,
        help="New results directory. Defaults to output/<dataset>_video_run in this repository.",
    )
    parser.add_argument("--objects", required=True, type=parse_classes,
                        help='JSON list such as [{"name":"floor","prompts":{"wooden floor":0.5}}]')
    parser.add_argument("--checkpoint", required=True, type=Path, help="Local SAM 3.1 multiplex checkpoint")
    parser.add_argument("--episodes", nargs="*", default=[], help="Episode directory names. Empty means all.")
    parser.add_argument("--cameras", nargs="+", default=["front_head"], help="Camera directory names")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0], help="GPU indices relative to visible CUDA devices")
    parser.add_argument("--batch-size", type=int, default=4, help="Initial frames per inference batch per GPU")
    parser.add_argument("--fps", type=int, default=20, help="Playback frames per second")
    parser.add_argument("--compile", action="store_true", help="Compile the detector's ViT backbone with torch.compile")
    parser.add_argument("--fa3", action="store_true", help="Enable FlashAttention 3 on compatible Hopper GPUs")
    args = parser.parse_args()

    source = args.dataset.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    default_output = Path(__file__).resolve().parents[1] / "output" / f"{source.name}_video_run"
    output = (args.output or default_output).expanduser().resolve()
    try:
        if output.exists():
            raise ValueError(f"Output already exists. Choose a new path: {output}")
        if not checkpoint.is_file():
            raise ValueError(f"Checkpoint is missing: {checkpoint}")
        if args.batch_size < 1 or min(args.gpus) < 0 or len(set(args.gpus)) != len(args.gpus):
            raise ValueError("Batch size must be positive and GPU indices must be unique and nonnegative.")
        if args.fps < 1:
            raise ValueError("FPS must be positive.")
        if not all(shutil.which(tool) for tool in ("ffmpeg", "ffprobe")):
            raise ValueError("FFmpeg and ffprobe are required.")
        frames = list_frames(source, args.episodes, args.cameras)
    except ValueError as error:
        parser.error(str(error))

    groups = defaultdict(list)
    for frame in frames:
        groups[(frame.episode, frame.camera)].append(frame)
    assignments = assign_groups(groups, args.gpus[:len(groups)])
    print(f"Segmenting {len(frames)} frames across {len(groups)} cameras on GPUs {list(assignments)}. "
          f"Output: {output}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".sam3-segment-", dir=output.parent))
    ctx = mp.get_context("spawn")
    reports = ctx.Queue()
    processes = []
    compile_cache = None
    try:
        if args.compile:
            # Keep concurrent workers off the same compiler cache.
            compile_cache = tempfile.TemporaryDirectory(prefix="sam3-compile-")
        for gpu, camera_groups in assignments.items():
            process = ctx.Process(
                name=f"sam3-gpu-{gpu}", target=run_worker,
                args=(gpu, camera_groups, args.objects, checkpoint, args.batch_size,
                      args.fps, stage, reports, args.compile, args.fa3,
                      Path(compile_cache.name) / str(gpu) if compile_cache is not None else None),
            )
            process.start()
            processes.append(process)
        results = collect_reports(processes, reports, groups)
        if output.exists():
            raise FileExistsError(f"Output was created by another process: {output}")
        stage.rename(output)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
        reports.close()
        if compile_cache is not None:
            compile_cache.cleanup()
        if stage.exists():
            shutil.rmtree(stage)
    print_reports(results, args.objects)
    print(f"Completed {len(frames)} frames. Results: {output}", flush=True)


if __name__ == "__main__":
    main()
