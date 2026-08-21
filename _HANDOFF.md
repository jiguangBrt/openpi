# OpenPI UR Ping-Pong Fine-Tuning Handoff

更新时间：2026-07-21（Asia/Shanghai）

## 当前目标与状态

目标是在当前 `uv` 环境中注册并训练 JAX + LoRA 配置 `pi05_ur_ping_pong`，使用本地 LeRobot 数据集：

```text
/mnt/reacher-fast/openpi_ur_pp_202607/lerobot_data/ur_pick_up_ping_pong
```

配置、数据加载、delta action、归一化统计和完整 batch 已经验证。**正式训练尚未启动**，当前停在启动训练之前。

## 已确认的训练配置

配置位于 `src/openpi/training/config.py`，名称为 `pi05_ur_ping_pong`。

```text
模型                    pi0.5（JAX）
PaliGemma variant       gemma_2b_lora（rank 16）
Action expert variant   gemma_300m_lora（rank 32）
action_horizon          10（数据为 10 Hz，对应未来 1 秒）
discrete_state_input    True
global batch size       8
num_train_steps         10,000
warmup_steps            1,000
peak_lr                 2.5e-5
decay_steps             10,000
decay_lr                2.5e-6
optimizer               AdamW，gradient clip 1.0
EMA                     关闭
save_interval           1,000
keep_period             5,000
seed                     42（TrainConfig 默认值）
```

使用 OpenPI 官方 `Pi0Config.get_freeze_filter()` 标准 LoRA 策略。它并非严格的 LoRA-only，而是：

```text
总参数                  3,403,421,456
可训练参数                466,957,072
其中 LoRA                  49,987,584
完整 SigLIP 视觉编码器     414,803,696
action/time heads            2,165,792
```

用户已经明确选择保留此官方标准 LoRA 策略，不额外冻结视觉编码器。

## 数据集信息

```text
LeRobot codebase version  v2.1
robot_type                 ur
episodes                   130
frames                     17,477
fps                        10
tasks                      1
images                     image + wrist_image，均为 224x224 RGB
state                      7 维（6 关节 + 夹爪）
actions                    7 维（6 关节 + 夹爪）
```

任务提示：

```text
Pick up the yellow ping-pong ball and place it in the white box.
```

数据中的 action 是绝对关节目标。当前变换将前 6 维转换为相对当前 state 的 delta，第 7 维夹爪保持绝对值。推理输出通过 `AbsoluteActions` 恢复为绝对关节目标。

## 已修改文件

### `src/openpi/training/config.py`

- 修复原草稿中重复导入 `libero_policy`、漏导入 `ur5_policy` 的错误。
- 将错误嵌套在 `LeRobotAlohaDataConfig` 内的 `LeRobotUR5DataConfig` 移到模块顶层。
- 在 `DataConfig` 中增加可选的 `repo_root`，用于明确指定本地 LeRobot 数据根目录。
- 添加 UR 数据 repack、UR policy transforms、前 6 维 delta/absolute action transforms。
- 注册并落实上方列出的 `pi05_ur_ping_pong` 超参数。
- `repo_id` 为 `ur_pick_up_ping_pong`，`asset_id` 默认同名，因此 norm stats 读写路径一致。

### `src/openpi/policies/ur5_policy.py`

新增 UR policy transform：

- `image` 映射到 `base_0_rgb`。
- `wrist_image` 映射到 `left_wrist_0_rgb`。
- 缺失的 `right_wrist_0_rgb` 使用黑图填充，mask 为 false。
- state 和 actions 交给通用模型 transform padding 到 32 维。
- 推理输出只保留前 7 维动作。

### `src/openpi/training/data_loader.py`

修改原因有两个：

1. 原始 OpenPI loader 只能根据 `repo_id` 从 `$HF_LEROBOT_HOME/<repo_id>` 查找数据，无法直接表达当前工作区中的本地数据路径。现在会把 `DataConfig.repo_root` 传给 `LeRobotDatasetMetadata` 和 `LeRobotDataset`，无需环境变量、缓存目录软链接或绝对路径伪装成 repo ID。
2. 当前 uv 环境使用 `datasets==3.6.0`，但数据集 Parquet metadata 由较新 Hugging Face `datasets` 写出，包含 `_type: "List"`。原始 loader 会报：

   ```text
   ValueError: Feature type 'List' not found
   ```

   新增 `_LocalLeRobotDataset`，从 LeRobot `meta/info.json` 构造显式 `datasets.Features` 后加载 Parquet，绕过不兼容的 Parquet metadata。

此兼容路径只在设置了本地 `repo_root` 时启用；远程 Hugging Face 数据集仍使用原始 LeRobot loader。没有改写任何原始 Parquet 文件。

### `uv.lock`

`uv.lock` 在本次工作开始前已经存在大规模未提交修改（主要是 PyPI 源切换到清华镜像）。本次没有修改或回退它。

## 归一化统计

由于开启了 delta action，不能复用数据集根目录中基于绝对 action 的旧统计量。已经用当前 uv 环境遍历全量数据重新计算统计量：

```text
assets/pi05_ur_ping_pong/ur_pick_up_ping_pong/norm_stats.json
```

该文件当前存在，但 `assets/` 被 `.gitignore` 忽略。在新工作区或删除 assets 后，需要重新运行：

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_ur_ping_pong
```

新的前 6 维 action 均值接近 0，夹爪统计保持绝对值；pi0.5 使用 q01/q99 进行 quantile normalization。

## 已完成测试

### 配置与静态检查

以下检查通过：

```text
config.get_config("pi05_ur_ping_pong")
git diff --check
ruff check
ruff format
python -m py_compile
```

当前 uv 环境没有安装 `pyright`，因此未执行 Pyright 类型检查。

### 单元测试

运行了相关 data loader 和 pi0 LoRA 测试（排除项目原有的远程真实数据测试）：

```text
8 passed, 1 deselected
```

命令：

```bash
HF_DATASETS_CACHE=/tmp/openpi-hf-datasets \
UV_CACHE_DIR=/tmp/uv-cache \
uv run --no-sync pytest -q \
  src/openpi/training/data_loader_test.py \
  src/openpi/models/pi0_test.py \
  -k 'not real'
```

### 本地数据与 delta action

- 成功读取全部 17,477 个本地样本。
- 单个样本 action chunk 为 `(10, 7)`。
- 数值验证 `delta_action[:6] == absolute_action[:6] - current_state[:6]`。
- 验证夹爪 action 保持绝对值。
- prompt 能从 `task_index` 正确注入。

### 完整训练 batch

成功执行：

```text
LeRobot load
-> prompt 注入
-> UR repack
-> delta action
-> quantile normalization
-> state tokenization
-> image resize
-> state/action padding
-> batch
```

输出：

```text
state              (8, 32), float32
actions            (8, 10, 32), float32
tokenized_prompt   (8, 200), int32
base image         (8, 224, 224, 3), float32
wrist image        (8, 224, 224, 3), float32
right wrist image  (8, 224, 224, 3), float32
```

相机 mask 中 base 和 left wrist 为 true，补齐的 right wrist 为 false。每个样本实际使用 51 个 prompt/state token，未超过 `max_token_len=200`。

### GPU/JAX 环境

通过沙箱外 `nvidia-smi` 和当前 uv 环境确认：

```text
8 x NVIDIA GeForce RTX 5090
每张显存 32,607 MiB
Driver 580.159.03
JAX backend: gpu
JAX device_count: 8
```

global batch 8 可被 8 张 GPU 整除，对应每卡 batch 1。默认 `fsdp_devices=1` 会使用 8 卡数据并行，每卡持有一份模型。

## 尚未完成/尚未验证

- 尚未启动正式训练。
- 尚未执行真实的第一个梯度更新。
- 尚未验证 checkpoint 下载和基础权重恢复。
- pi0.5 base checkpoint 当前不在本地缓存中；首次训练将从以下地址下载：

  ```text
  gs://openpi-assets/checkpoints/pi05_base/params
  ```

- PaliGemma tokenizer 已存在于 `/home/user/.cache/openpi/big_vision/paligemma_tokenizer.model`。
- W&B 是否启用尚未最终决定；`TrainConfig` 默认启用。

## 下一步：正式训练

启用 W&B：

```bash
cd /mnt/reacher-fast/openpi_ur_pp_202607/repo

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_ur_ping_pong \
  --exp-name=pi05_finetune_ur_lora \
  --overwrite
```

禁用 W&B：

```bash
cd /mnt/reacher-fast/openpi_ur_pp_202607/repo

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_ur_ping_pong \
  --exp-name=pi05_finetune_ur_lora \
  --overwrite \
  --no-wandb-enabled
```

启动后应重点观察：

- 8 张 GPU 是否均有显存占用和计算负载。
- 首次 checkpoint 下载是否成功。
- 初始化日志中的 batch shape 是否与上方一致。
- loss、grad norm 是否为有限值。
- 每卡显存是否稳定低于 32 GB。
- 1,000/5,000/最终 checkpoint 是否按预期保存。

## 当前 Git 工作区

写入 handoff 前的状态为：

```text
 M src/openpi/training/config.py
 M src/openpi/training/data_loader.py
 M uv.lock
?? src/openpi/policies/ur5_policy.py
```

加上本文件后，`_HANDOFF.md` 也应为未跟踪文件。所有修改均未提交。
