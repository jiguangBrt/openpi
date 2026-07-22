# `pi05_ur_ping_pong` 训练指令

所有命令均在项目目录执行：

```bash
cd /mnt/reacher-fast/openpi_ur_pp_202607/repo
```

当前配置使用本地 `ur_pick_up_ping_pong` 数据集，以 π₀.₅ 基础权重进行双侧 LoRA 微调。训练共 10,000 step，global batch size 为 8，每 1,000 step 保存一次 checkpoint。

## 启动前快速检查

```bash
test -d ../lerobot_data/ur_pick_up_ping_pong && echo "dataset: OK"
test -f assets/pi05_ur_ping_pong/ur_pick_up_ping_pong/norm_stats.json && echo "norm stats: OK"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
```

如果数据或 norm stats 检查没有输出 `OK`，先不要启动训练。

## 首次训练

建议在 tmux 中运行，避免 SSH 断开后训练退出：

```bash
tmux new -s pi05_train
```

进入 tmux 后执行：

```bash
cd /mnt/reacher-fast/openpi_ur_pp_202607/repo

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run --no-sync scripts/train.py pi05_ur_ping_pong \
  --exp-name=pi05_finetune_ur_lora \
  --no-wandb-enabled
```

首次运行会自动下载尚未缓存的 `pi05_base` 权重。不要在首次启动时添加 `--overwrite`，这样同名实验已存在时会安全退出。

tmux 常用操作：

```text
Ctrl-b d                         # 退出会话，训练继续运行
tmux attach -t pi05_train        # 重新进入会话
```

## 停止与恢复

在训练终端按 `Ctrl-C` 安全停止。训练会从最近一次已保存的 checkpoint 恢复，不会在中断时额外保存。

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run --no-sync scripts/train.py pi05_ur_ping_pong \
  --exp-name=pi05_finetune_ur_lora \
  --resume \
  --no-wandb-enabled
```

`--resume` 和 `--overwrite` 不要同时使用。

## 开始新实验

保留旧 checkpoint 时，使用新的实验名：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run --no-sync scripts/train.py pi05_ur_ping_pong \
  --exp-name=pi05_finetune_ur_lora_v2 \
  --no-wandb-enabled
```

只有确认旧实验不再需要时，才使用原实验名加 `--overwrite`：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run --no-sync scripts/train.py pi05_ur_ping_pong \
  --exp-name=pi05_finetune_ur_lora \
  --overwrite \
  --no-wandb-enabled
```

## 查看运行状态

```bash
pgrep -af 'scripts/train.py pi05_ur_ping_pong'
watch -n 2 nvidia-smi
find checkpoints/pi05_ur_ping_pong/pi05_finetune_ur_lora \
  -maxdepth 2 -type d -print | sort
```

10,000 次更新的最终 checkpoint 目录预计为 `9999`。按当前保留策略，训练完成后重点使用 `5000` 和 `9999`。

## 数据变化后重新计算 norm stats

仅当数据、state/action 定义或 transform 发生变化时执行：

```bash
uv run --no-sync scripts/compute_norm_stats.py --config-name pi05_ur_ping_pong
```

输出文件：

```text
assets/pi05_ur_ping_pong/ur_pick_up_ping_pong/norm_stats.json
```

## 使用训练结果启动策略服务

下面的命令只向进程暴露物理 GPU 7（机器上的第 8 张卡），加载最终 checkpoint，并开启推理输入/输出录制：

```bash
cd /mnt/reacher-fast/openpi_ur_pp_202607/repo

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=7 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
uv run --no-sync scripts/serve_policy.py \
  --port=8000 \
  --record \
  --default-prompt="Pick up the yellow ping-pong ball and place it in the white box." \
  policy:checkpoint \
  --policy.config=pi05_ur_ping_pong \
  --policy.dir=checkpoints/pi05_ur_ping_pong/pi05_finetune_ur_lora/9999
```

如果使用中间 checkpoint，请把末尾的 `9999` 替换为实际 step。设置 `CUDA_VISIBLE_DEVICES=7` 后，JAX 可能把这张卡显示为进程内的 `CudaDevice(id=0)`；这是物理 GPU 7 到逻辑 GPU 0 的正常映射。

服务成功启动后会输出：

```text
Dumping policy records to: policy_records
server listening on 0.0.0.0:8000
```

可在服务端检查健康状态：

```bash
curl http://127.0.0.1:8000/healthz
```

正常返回 `OK`。策略服务只负责根据 observation 预测 action chunk，不会自行连接或控制机器人。

### 客户端安装与调用示例

如果机器人客户端使用独立的 Python 环境，先安装轻量客户端包：

```bash
cd /mnt/reacher-fast/openpi_ur_pp_202607/repo/packages/openpi-client
pip install -e .
```

以下代码展示一次标准推理调用。`read_robot_state()`、`read_base_camera()` 和 `read_wrist_camera()` 代表机器人程序已有的状态及相机读取函数，需要替换为实际实现：

```python
import numpy as np

from openpi_client import image_tools
from openpi_client import websocket_client_policy


def prepare_image(image: np.ndarray) -> np.ndarray:
    image = image_tools.convert_to_uint8(np.asarray(image))
    return image_tools.resize_with_pad(image, 224, 224)


# 同机调用用 127.0.0.1；远程调用改为推理服务器的真实局域网 IP。
client = websocket_client_policy.WebsocketClientPolicy(
    host="127.0.0.1",
    port=8000,
)

observation = {
    "state": np.asarray(read_robot_state(), dtype=np.float32),
    "image": prepare_image(read_base_camera()),
    "wrist_image": prepare_image(read_wrist_camera()),
    "prompt": "Pick up the yellow ping-pong ball and place it in the white box.",
}

result = client.infer(observation)
action_chunk = np.asarray(result["actions"])

assert action_chunk.shape == (10, 7)
print("action chunk:", action_chunk)
print("policy timing:", result.get("policy_timing"))
print("server timing:", result.get("server_timing"))
```

启动服务时已经设置了 `--default-prompt`，因此客户端可以省略 `prompt`；显式传入时，客户端提供的 prompt 优先。

### 调用规范

输入 observation 的字段如下：

| 字段 | 类型和形状 | 必需 | 说明 |
| --- | --- | --- | --- |
| `state` | `np.ndarray`, `(7,)` | 是 | 当前机器人状态，前 6 维为机械臂关节，第 7 维为夹爪；顺序、单位和数值范围必须与训练数据一致。 |
| `image` | `np.ndarray`, `(224, 224, 3)`, `uint8` | 是 | 基座/场景相机 RGB 图像，采用 HWC 排列。 |
| `wrist_image` | `np.ndarray`, `(224, 224, 3)`, `uint8` | 是 | 腕部相机 RGB 图像，采用 HWC 排列。 |
| `prompt` | `str` | 否 | 任务指令；省略时使用服务端的 `--default-prompt`。 |

图像应是 RGB 而不是 BGR。如果相机通过 OpenCV 读取 BGR 图像，应先转换成 RGB。浮点图像传给 `convert_to_uint8` 时应位于 `[0, 1]` 范围。

成功响应至少包含：

| 字段 | 类型和形状 | 说明 |
| --- | --- | --- |
| `actions` | `np.ndarray`, `(10, 7)` | 未来 10 个控制步的动作。前 6 维是恢复后的绝对关节目标，第 7 维是绝对夹爪目标。 |
| `policy_timing` | `dict` | 模型侧推理耗时，包含 `infer_ms`。 |
| `server_timing` | `dict` | 服务端请求耗时信息。 |

训练数据频率为 10 Hz，`action_horizon=10`，所以完整 action chunk 覆盖约 1 秒。机器人端可以完整开环执行 10 步，也可以只执行前若干步后重新请求；无论采用哪种方式，都必须在机器人端进行关节限位、速度限制、碰撞检测和急停保护。

### Record 录制文件

使用上面的 `--record` 参数后，录制目录相对于启动服务时的当前工作目录。按本文命令启动时，绝对路径为：

```text
/mnt/reacher-fast/openpi_ur_pp_202607/repo/policy_records/
```

每成功完成一次 `client.infer()`，会生成一个文件：

```text
policy_records/step_0.npy
policy_records/step_1.npy
policy_records/step_2.npy
...
```

每个文件保存该次请求的原始 observation 和最终策略输出。可以这样读取：

```bash
cd /mnt/reacher-fast/openpi_ur_pp_202607/repo

python - <<'PY'
import numpy as np

record = np.load("policy_records/step_0.npy", allow_pickle=True).item()

print("record keys:", record.keys())
print("state:", record["inputs/state"])
print("action shape:", record["outputs/actions"].shape)
print("actions:", record["outputs/actions"])
PY
```

Record 记录的是每次模型请求的输入和输出，不是相机视频，也不会记录 action chunk 中间每个控制步的真实机器人反馈。如果客户端每 10 个控制步请求一次，目录中只会产生一个对应的 `step_N.npy`。

服务进程每次启动时，record 编号都会从 `step_0.npy` 重新开始，已有同名文件会被覆盖。需要保留旧记录时，应在重启服务前先将整个 `policy_records` 目录改名或移动到归档位置。

## 常见问题

- `FileExistsError`：继续旧训练用 `--resume`；开始新实验就更换 `--exp-name`。
- 找不到 norm stats：运行上面的 norm stats 计算命令。
- JAX 只看到 CPU 或 GPU 数量不对：检查 `echo "$CUDA_VISIBLE_DEVICES"` 和 `nvidia-smi`。当前 global batch 8 要求可见 GPU 数量能整除 8。
- 显存不足：先关闭其他 GPU 任务；仍然不足时可把 `XLA_PYTHON_CLIENT_MEM_FRACTION` 降到 `0.85`。
- 基础权重下载失败：检查网络和 `~/.cache/openpi` 的空间、权限。
