---
name: wr-fold-short-jeans-sam31
description: 在 Slurm 上直接读取折短裤 LeRobot v2.1 数据，用 SAM 3.1 分割五类物体并修复 front_head 桌面和地板掩码，发布掩码索引与检出率。适用于制作或复查 UnifoLM-WLA 背景换色训练数据。
---

# 折短裤数据的 SAM 3.1 分割与修复

用户尚未给路径时，先询问 LeRobot 数据集根目录的绝对路径。默认直接读取 `meta/info.json` 和 `meta/episodes.jsonl`。根目录有 `train/` 时只扫描该目录，也可传入单个 LeRobot 子数据集。扫描分类目录和指向 LeRobot 子数据集的软链接，按路径排序并按真实源目录去重。跳过 `prepared`、`eval` 和名称带 `_eval` 的子数据集。

检查 LeRobot v2.1、`front_head` 视频 20 FPS、视频帧数与 Parquet 行数一致，以及 `frame_index` 从零连续递增。过滤后的 Parquet 时间戳可在 20 FPS 网格上跳号，须严格递增。视频按 `frame_index` 对齐。

## 运行

在 Slurm 登录节点进入本 Skill 所在的 SAM3 仓库。使用同级 `../conda_envs/sam31/bin/python` 和 `../weight/sam3.1_multiplex.pt`。确认 `import av, pyarrow.parquet` 成功。现有环境可用 `--deps ../.sam3-lerobot-full.00HPdy0B/deps` 加载 PyAV 和 PyArrow，使用前检查该目录仍可读。

将 `dataset` 设为用户给出的根目录。

```bash
../conda_envs/sam31/bin/python -B scripts/run_segment_lerobot.py \
  --no-media --dataset-root "$dataset" \
  --deps ../.sam3-lerobot-full.00HPdy0B/deps --output "$PWD/output/$(basename -- "$dataset")"
```

使用 `--shards`、`--partition`、`--nodelist` 调整 Slurm 资源。小样使用 `--ordinals 0 --shards 1` 和独立报告目录。旧清单可用 `--manifest <清单路径>` 显式指定，沿用清单模式的交付。使用 `--reuse-segmentation <分割目录>` 可复用已完成的首轮分割，目录须有 `plan.json`、`completed.json` 和首轮 Parquet。无媒体运行还需要 `labels.npz`，旧媒体运行使用五类掩码视频。历史掩码沿用原提示词。

## 交付与核验

- 分割桌面、地板、衣服、短裤和机械臂。衣服提示词掩码取并集，其中 `purple shirt` 阈值为 `0.50`。对桌地空掩码或异常重叠片段做双向跟踪。无法可靠修正的帧写空桌地掩码，换色训练使用原图。
- 在数据集根目录的 `background_masks_tracked/` 下按分类和子数据集发布六列 Parquet，包含 `frame_index` 和五类无损 PNG 掩码。直接扫描模式同时发布 `mask_index.json`，记录源数据集、集号、Parquet、视频、帧数和掩码路径。小样或已有产物时使用独立运行子目录，以发布日志为准。单独要求首轮分割时使用 `--segmentation-only`。
- 等待发布完成，核对 `output/<dataset_name>/detection_rates.json` 的 episode 数、帧数和掩码索引路径。报告按集及全量统计 `raw`、`before` 和 `after`。首轮 `before` 的桌地已扣除前景，三类前景沿用首轮结果。发布完成后清理本次临时掩码和作业目录。失败时检查临时目录的 `logs/` 、提交时打印的发布失败日志及 `sacct`。

## WLA 换色训练

用训练选集自己的清单和统计。运行 WLA 仓库中的合并命令，为选中的 episode 附加掩码路径。

```bash
python scripts/merge_background_masks.py \
  --manifest /path/to/prepared/manifest.json \
  --mask-index /path/to/background_masks_tracked/mask_index.json \
  --output /path/to/prepared/manifest_background_masks_tracked.json
```

开启换色并将 `background_recolor_manifest_path` 指向合并后的清单。训练按帧读取桌面和地板掩码。不同选集可以使用同一份掩码索引。
