# UR 真机数据 → 远程训练 → 仿真评估 → UR5e 真机部署：分步操作指南

> 本文是一份**面向操作**的端到端手册:从同事采集好的 LeRobot UR 数据集出发,在远程 GPU 机器上微调 π0.5,做仿真评估,最后部署到 UR5e 真机。
>
> 配套阅读(理论,本文不重复):[train_custom_pi0.md](train_custom_pi0.md)(配置体系/三层变换/服务启动)、[data_flow.md](data_flow.md)(一条数据流串讲)、[architecture.md](architecture.md)(逐文件目录详解)。
>
> 所有源码引用均可点击跳转(相对路径基于本 `tutorial/` 目录,指向 `../`)。

---

## 0. 先看清三个事实(避免踩坑)

在动手前,你必须知道 openpi 对 UR 的支持**现状**,否则会按"有现成例子"的预期走弯路:

1. **`examples/ur5/` 只是文档模板,不是可运行例子。** 整个目录只有一个 [README.md](../examples/ur5/README.md),里面用代码片段教你**自己写** `UR5Inputs`/`UR5Outputs`/`LeRobotUR5DataConfig`/`TrainConfig`。没有 `env.py`、没有 `main.py`、没有数据转换脚本、**`src/openpi/training/config.py` 里也没有注册任何 `ur5` config**(已 grep 确认)。可运行的完整例子是 `examples/libero/`、`examples/aloha_sim/`、`examples/droid/` —— UR 要照着它们自己搭。
2. **openpi 不带 UR 仿真环境。** `examples/libero/` 用 LIBERO 仿真,`examples/aloha_sim/` 用 MuJoCo,但 UR 没有。所以"在仿真里评估"这一步需要**你自己搭一个 UR 仿真**(本文第五节给方案),或临时借用 libero 仿真验证"训练→推理→rollout"流程是否打通。
3. **openpi 不带 UR 真机驱动。** 仓库里没有任何 `rtde`/`ur_rtde`/`ros` 代码。真机部署的机器人侧循环(读关节/夹爪、拍照、发观测、收动作、下发给 RTDE)需要**你自己写**,用 [openpi-client](../packages/openpi-client/) 的 `WebsocketClientPolicy` + `ActionChunkBroker` 做推理侧,模板是 [examples/simple_client/main.py](../examples/simple_client/main.py) 和 [examples/libero/main.py](../examples/libero/main.py)。

> 关于机器人型号:你说的是"UR7e",但 Universal Robots 没有 UR7e 这个型号(有 UR3e/UR5e/UR10e/UR16e/UR20/UR30)。openpi 的 base 资产里有 `asset_id="ur5e"` 的预训练归一化统计量([../docs/norm_stats.md](../docs/norm_stats.md) 描述为"6-DoF UR5e + Robotiq 2F-85"),你本地的数据集 `robot_type` 也标为 `"ur"`、state/action 都是 6 关节+1 夹爪=7 维。**本文按 UR5e 写**;若实际是 UR10e 等,由于动作在**关节空间**,迁移照样成立,只是关节限位/运动学不同,不影响训练流程。

---

## 阶段总览

| 阶段 | 在哪做 | 产出 |
|---|---|---|
| A. 环境与远程探测 | 本地 + 远程 | 确认 GPU/后端/磁盘,选定 JAX 或 PyTorch |
| B. 数据分析与校验 | 本地 | 确认 schema、动作绝对/增量、fps、数据质量 |
| C. 注册 `pi05_ur_ping_pong` 配置 | 代码(本地改、同步到远程) | 可被 `get_config()` 找到的 TrainConfig |
| D. 计算 norm_stats | 远程 | `assets/pi05_ur_ping_pong/<repo_id>/` 归一化统计 |
| E. 远程训练 | 远程 GPU | `checkpoints/pi05_ur_ping_pong/<exp>/<step>/` |
| F. 仿真评估 | 本地或远程 | 验证策略能 rollout 出合理动作 |
| G. 真机部署 UR5e | 真机旁机器 + GPU 机器 | 闭环跑通 |

---

## A. 环境与远程机器探测

### A.1 本地环境(已知,备忘)

- 仓库实际代码在 `/home/jh/OpenPI_UR/openpi/`(外层 `/home/jh/OpenPI_UR/` 还放了 `datasets/`、`tutorial/`)。
- 本地 RTX 3060 6GB,**低于 openpi 8GB 推理下限**,只用于读代码/分析数据,**不跑模型**([OpenPI README](../README.md))。
- 依赖用 **uv** 管理:`GIT_LFS_SKIP_SMUDGE=1 uv sync`;跑脚本一律 `uv run ...`。
- git over https 需代理:`git config --global http.proxy http://127.0.0.1:17891`(+ `https.proxy`)。PyPI 不需要代理。

### A.2 远程机器实测结果(已探测,2026-07-20)

远程是 `user@user-virtual-machine`(与同事训练 `ur_0706_130p_lora` 的机器**不是同一台**)。实测:

| 项 | 结果 | 含义 |
|---|---|---|
| GPU | **8× RTX 5090,compute_cap 12.0(Blackwell sm_120)**,全空闲 | 算力充足;但 Blackwell 上 JAX 能否跑需实测(见 A.4) |
| 驱动/CUDA | Driver 580,CUDA 13.0 | CUDA 13 驱动对 CUDA 12 运行时向下兼容 |
| github | `git ls-remote` 成功 | ✅ **直连可达,不需要代理**(和本地不同) |
| uv | `/home/user/.local/bin/uv` 已装 | ✅ |
| uv 缓存 | `~/.cache/uv` = 0 | 没下过 wheel,`uv sync` 要全量下载 |
| openpi | **未 clone**;`~/.cache/openpi` 仅 4.1M | base 权重没下过,需下载 |
| gsutil/gcloud | **未装** | 下 base 权重(`gs://openpi-assets`)需装 gsutil 或靠 gcsfs 回退 |
| 现有 conda 环境 | 有 `openpi_env` 但**空的**;`lyra2` 里有 **torch 2.7.1+cu128** | **没有任何环境装了 jax/jaxlib/orbax/flax** → JAX 必须靠 `uv sync` 装,无法复用现有环境 |
| 磁盘 | `/`(sda2)只剩 **27G**;但 `/mnt/reacher-fast` 剩 **139G(本地 SSD)**、`/mnt/iscsi-3.8t` 剩 **1.1T**、`/mnt/data_nfs2` 剩 3.1T | ✅ **硬盘可解**:项目放大盘,别放家目录 |

**关键判断:**
1. **硬盘不是阻碍**——把 clone+venv+活跃 checkpoint 放 `/mnt/reacher-fast`(本地 SSD,IO 快),归档放 `/mnt/iscsi-3.8t`。家目录 `/` 27G 会被 venv(~15-20GB)撑爆,绝对不要用。
2. **PyTorch 路线已基本确认可行**——`lyra2` 环境里就有 torch 2.7.1+cu128(cu128 支持 sm_120)在跑。openpi 锁定 `torch==2.7.1`,PyPI 默认 wheel 是 cu128 构建,能在 RTX 5090 上跑。
3. **JAX 路线待实测**——openpi 锁定 `jax[cuda12]==0.5.3`,jaxlib 是否带 sm_120 PTX 要 `uv sync` 后跑 GPU 实算才知道(光 `jax.devices()` 出 GPU 不算数)。
4. **转换 JAX base → PyTorch 需要装 JAX,但只在 CPU 上读参数**——即便 JAX GPU 不可用,转换也能在 CPU 上跑(慢,一次性)。

### A.3 选后端:JAX 还是 PyTorch

| | JAX(`scripts/train.py`) | PyTorch(`scripts/train_pytorch.py`) |
|---|---|---|
| openpi 主路径 | ✅ 是 | 镜像实现 |
| LoRA 微调 | ✅ 支持 | ❌ **不支持**([../README.md](../README.md) 明确) |
| 在 RTX 5090 上 | **待实测**(jax 0.5.3 / sm_120) | ✅ 基本确认可行(torch 2.7.1+cu128) |
| 多卡 | FSDP(`--fsdp_devices=N`) | DDP(`torchrun --standalone --nproc_per_node=N`) |
| base 权重来源 | 直接 `gs://openpi-assets/checkpoints/pi05_base/params` | 需先用 [convert_jax_model_to_pytorch.py](../examples/convert_jax_model_to_pytorch.py) 转 `model.safetensors`,设 `pytorch_weight_path` |
| 推理后端自动检测 | ckpt 目录无 `model.safetensors` → JAX | 有 `model.safetensors` → PyTorch([policy_config.py](../src/openpi/policies/policy_config.py)) |

**决策树(按 A.4 实测结果走):**
- `uv sync` 后跑 JAX GPU 实算 → **通过** → 走 **JAX + LoRA**(和同事 `ur_0706_130p_lora` 同路线,可直接借鉴他的 config)。
- JAX 实算 → **失败**(`no kernel image` 等) → 走 **PyTorch 全参微调**(无 LoRA,但 8×32GB 全参微调 π0.5 ~3B 参数绰绰有余)。需先在 CPU 上把 pi05_base 转成 PyTorch 格式。

下文 E 节按两种后端分别给命令,实测后取其一。

### A.4 磁盘最终决策 + 环境搭建 + 后端实测(远程,会下载 ~20GB)

> 探测已完。远程是**共享 `user` 账号**(所有同事都用 `user` 登录),所以 `ls -la` 看属主区分不了人。正确做法:**在一个大盘里建一个名字独一无二、自带项目标识的子目录,把整个部署关在里面**,绝不往外面写——这样既不碰同事文件,将来 `rm -rf` 这一个目录就等于完全卸载。

**磁盘实测结论(df + 可写性,2026-07-20):**

| 挂载点 | 总量/空闲 | 可写 | 判断 |
|---|---|---|---|
| `/mnt/reacher-fast` | 295G / **139G** | ✅ 本地 SSD | **首选**:IO 快,放 repo/`.venv`/缓存/权重/数据集 |
| `/mnt/iscsi-3.8t` | 3.8T / 1.1T | ✅ iSCSI 网络盘 | 兜底:放训练 checkpoint 归档(慢,不放开发布署) |
| `/mnt/data_nfs2` | 8T / 3.1T | ✅ NFS | 备用大空间 |
| `/mnt/data800` | 787G / 64G | ✅ 但挂着 `xiang@...ai_share` | **别碰**,别人的数据 |
| `/mnt/data_mount` | 1.8T / 36G | ✅ 但 98% 满 | 排除 |
| `data13T/data1600/data2t/data801/802/disks/fsd_data/iscsi/sdd` | 98G / **27G** | 多数 no-write | **全是根盘 `/`**,27G 会爆,排除 |

**关键约束(共享账号):**
1. **不要改 `~/.bashrc`**——会影响所有同事。环境变量写进项目里的 `env.sh`,每次开终端 `source env.sh`,只影响当前会话。
2. **不要在 `/mnt/reacher-fast` 根目录直接 clone**(已被别人用了 141G),一定进自己的子目录。
3. `reacher-fast` 虽然是 `user` 属主,但这台机器同事没训练过(和同事那台不是同一台),撞车风险低;仍用独特名字 + 存在性检查双保险。

**自包含布局(一个文件夹 = 整个部署):**
```
/mnt/reacher-fast/openpi_ur_pp_202607/
├── env.sh          ← 只你 source,绝不改共享 ~/.bashrc
├── repo/           ← openpi 仓库 + .venv(uv sync 自动建)
├── uv-cache/       ← uv 依赖缓存(别污染共享 home 的 ~/.cache)
├── cache/          ← 基础权重 + norm_stats (OPENPI_DATA_HOME)
├── lerobot_data/   ← 你的数据集 (HF_LEROBOT_HOME)
└── checkpoints/    ← 训练产出
```

```bash
# 1. 建专属目录(带存在性保护,撞名就停)
export OPENPI_HOME=/mnt/reacher-fast/openpi_ur_pp_202607
[ -e "$OPENPI_HOME" ] && { echo "目录已存在,换个名字!"; } || mkdir -p "$OPENPI_HOME" && echo "CREATED $OPENPI_HOME"

# 2. 写 env.sh(环境变量集中在这里,不动 ~/.bashrc)
cat > $OPENPI_HOME/env.sh <<'EOF'
export OPENPI_HOME=/mnt/reacher-fast/openpi_ur_pp_202607
export OPENPI_DATA_HOME=$OPENPI_HOME/cache        # 基础权重、norm_stats
export UV_CACHE_DIR=$OPENPI_HOME/uv-cache         # uv 缓存,别污染共享 home
export HF_LEROBOT_HOME=$OPENPI_HOME/lerobot_data  # 本地数据集
mkdir -p $OPENPI_DATA_HOME $UV_CACHE_DIR $HF_LEROBOT_HOME $OPENPI_HOME/checkpoints
cd $OPENPI_HOME/repo
EOF

# 3. clone + uv sync(放本地 SSD,约 15-20GB)
source $OPENPI_HOME/env.sh
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Physical-Intelligence/openpi.git $OPENPI_HOME/repo
cd $OPENPI_HOME/repo
GIT_LFS_SKIP_SMUDGE=1 uv sync                    # 装 jax[cuda12]==0.5.3 + torch==2.7.1

# 4. JAX GPU 实算(Blackwell sm_120 关键测试)
uv run python -c "import jax; print('JAX_DEVICES', jax.devices()); import jax.numpy as jnp; x=jnp.ones((1,1,2048,2048),dtype=jnp.bfloat16); print('JAX_MATMUL_OK', float(jnp.sum(x@x)))"

# 5. PyTorch GPU 实算(兜底)
uv run python -c "import torch; x=torch.ones((2048,2048),dtype=torch.bfloat16,device='cuda'); print('TORCH_CUDA', torch.cuda.get_device_name(0), 'count', torch.cuda.device_count()); print('TORCH_MATMUL_OK', float((x@x).sum()))"
```

判读:`JAX_MATMUL_OK` 出数字 → **JAX + LoRA** 路线(和同事同);JAX 报 `no kernel image`/GPU 错 → **PyTorch 全参**路线。`TORCH_MATMUL_OK` 应该都会过。

> **实测结论(2026-07-21):JAX 路线确认通过。** `uv run`(在 `repo/` 下)跑出 `JAX 0.5.3` / `JAX_DEVICES [CudaDevice(id=0..7)]` / `JAX_MATMUL_OK 8589934592.0`;torch 同样 `TORCH_MATMUL_OK 8589934592.0`、`count 8`。**后端定为 JAX + LoRA**(和同事 `ur_0706_130p_lora` 同路线,可直接借鉴 config)。下文 C/D/E 取 JAX 分支。
>
> 踩坑备忘:`uv run` 必须在 `$OPENPI_HOME/repo` 下执行(那里有 `pyproject.toml`),否则 uv 找不到项目 venv、回落到 conda base(无 jax)。`env.sh` 里的 `cd $OPENPI_HOME/repo` 就是为此,每次开终端先 `source $OPENPI_HOME/env.sh`。验证用 `uv run python -c "import sys;print(sys.executable)"`,应指向 `.../repo/.venv/bin/python`。

> 之后每开新终端先 `source $OPENPI_HOME/env.sh`。后续(base 权重用 gcsfs 拉、数据集拷过来、注册 `pi05_ur_ping_pong`、训练)依赖后端结论,见 C/D/E 节。

> ⚠️ 远程 8 张卡若被别人占用(`nvidia-smi` 看显存),训练前先打招呼/换卡。`CUDA_VISIBLE_DEVICES` 选空闲卡。
> ⚠️ base 权重下载(`gs://openpi-assets`)需 gsutil;未装则装 Google Cloud CLI,或让 openpi 回退到 gcsfs(`uv pip install gcsfs`,公共桶可能需匿名访问)。见 D 节。

---

## B. 数据分析与校验(本地做)

你的数据集在 `/home/jh/OpenPI_UR/datasets/ur_pick_up_ping_pong/`(LeRobot v2.1)。这一步的目标:确认 schema 与你将写的变换/config 对得上,发现数据质量问题。

### B.1 已知事实(我已读取 meta/info.json)

| 项 | 值 |
|---|---|
| `robot_type` | `"ur"` |
| `total_episodes` / `total_frames` | 130 / 17477 |
| `fps` | **10** |
| 任务数 | 1(`"Pick up the yellow ping-pong ball and place it in the white box."`) |
| `image` | 外相机,[224,224,3],raw parquet(非视频) |
| `wrist_image` | 腕部相机,[224,224,3] |
| `state` | float32[7] = joint_0..5 + gripper(**已拼接**) |
| `actions` | float32[7] = joint_0..5 + gripper(**与 state 同轴**) |
| 已有 `norm_stats.json` | ✅ 含 mean/std/**q01/q99**(openpi quantile 格式,见 D 节说明) |

### B.2 用 lerobot 库做分析(在本地 openpi 目录跑)

仓库没有专门的"数据集检查工具",直接用 lerobot 库([已装在 .venv](../.venv/lib/python3.11/site-packages/lerobot/)):

```bash
cd /home/jh/OpenPI_UR/openpi
uv run python - <<'PY'
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

ROOT = "/home/jh/OpenPI_UR/datasets/ur_pick_up_ping_pong"
# 用一个假 repo_id + root 指向本地目录
meta = LeRobotDatasetMetadata("local/ur_pick_up_ping_pong", root=ROOT)
print("fps", meta.info["fps"])
print("episodes", meta.info["total_episodes"], "frames", meta.info["total_frames"])
print("features", list(meta.info["features"].keys()))
print("tasks", meta.tasks[:3])

ds = LeRobotDataset("local/ur_pick_up_ping_pong", root=ROOT)
s = ds[0]
for k, v in s.items():
    print(k, type(v).__name__, getattr(v, "shape", None), getattr(v, "dtype", None))

# 取若干 episode 的 state/actions 看范围、看哪些关节在动
import random
n = len(meta.episodes)
idxs = random.sample(range(n), 10) if n > 10 else list(range(n))
all_state, all_act = [], []
for i in idxs:
    ep_len = meta.episodes[i]["length"]
    for t in range(0, ep_len, 20):  # 抽样
        sample = ds[i * 0 + t]  # 注意:ds[t] 是全局 frame index,按需调整
# 更稳的做法:直接读 parquet
import pathlib, pyarrow.parquet as pq
tbl = pq.read_table(pathlib.Path(ROOT)/"data/chunk-000/episode_000000.parquet")
print(tbl.column_names)
st = tbl.column("state").to_pandas().tolist()
st = np.array([list(x) for x in st])
print("state shape", st.shape)
print("state per-dim min", st.min(0).round(3))
print("state per-dim max", st.max(0).round(3))
print("state per-dim std", st.std(0).round(4))   # 重点看哪些关节 std≈0
PY
```

### B.3 数据分析要回答的 5 个问题(直接决定 config 怎么写)

1. **动作是绝对还是增量?** 你的数据 `actions` 与 `state` 同轴、且 `min/max` 几乎一致(从 `episodes_stats.jsonl` 看 state 和 actions 的 min/max 数值几乎相同)→ **动作是绝对关节位置**。→ config 里**必须**加 `DeltaActions`(输入侧把绝对转增量) + `AbsoluteActions`(输出侧把增量还原回绝对),mask 用 `make_bool_mask(6, -1)`(前 6 关节转增量,夹爪维保持绝对)。这与 [examples/ur5/README.md](../examples/ur5/README.md) 模板一致。
2. **state 是否已拼接?** 你的 `state` 已是 7 维(joints+gripper 拼好),**不是** README 模板里 `UR5Inputs` 假设的 `joints`/`gripper` 分离字段。→ 你写的 `UR5Inputs` 要直接用 `data["state"]`,**不要** `np.concatenate([joints, gripper])`。这是模板必须改的地方。
3. **图像 key 名?** 你的数据是 `image`/`wrist_image`,而 README 模板的 `UR5Inputs` 读 `base_rgb`/`wrist_rgb`。→ repack 里把 `image→base_rgb`、`wrist_image→wrist_rgb`(或直接在 Inputs 里读 `image`/`wrist_image`,二选一,保持训练/推理一致)。
4. **fps 与 action_horizon。** 数据 fps=10。`pi05_libero` 在 10fps 下用 `action_horizon=10`(1 秒前瞻)。UR5e 真机按 [norm_stats.md](../docs/norm_stats.md) 是 20Hz 控制 —— **数据采集频率(10)与真机控制频率(20)不一致,要核实**:
   - 若数据是 10Hz 采集、真机也按 10Hz 下发 → action_horizon 取 10 左右。
   - 若数据是 20Hz 采集被降采样到 10 → 需明确真机下发频率,可能要 action_horizon=20。
   - **这步必须问清同事**:数据实际采集频率、真机打算用多少 Hz 下发。频率不匹配会让策略在真机上"动作播放速度"错乱。
5. **数据质量。** 从 `episodes_stats.jsonl` 看,joint_2(std≈0.0002)和 joint_5(std≈0.00005)几乎不动 → 该任务只用到部分关节。这是正常的(任务决定),但要在分析里确认:这些关节是真不动,还是采集时卡死(看原始图像/视频回放)。同时看是否有异常 episode(长度异常、state 跳变)。

> **检查点 B**:你能用一句话回答上面 5 个问题,并画出"数据集字段 → repack → UR5Inputs → DeltaActions → Normalize → model_transforms"的对应关系。否则别进 C。

---

## C. 注册 `pi05_ur_ping_pong` 配置

> ⚠️ 先问同事要他们已经用过的 `pi05_ur_ping_pong` config diff(`ur_0706_130p_lora` 那次训练用的)。**有现成的就直接用**,下面是"没有时要自己写"的模板。

### C.1 三块代码(放进 `src/openpi/`)

照 [examples/ur5/README.md](../examples/ur5/README.md) 模板,但**按 B.3 的发现改成 pi05 + 适配你的真实 schema**。

**① 平台变换** —— 新建 `src/openpi/policies/ur5_policy.py`(参考 [libero_policy.py](../src/openpi/policies/libero_policy.py) 逐行注释):

> ⚠️ 文件顶部必须带下面这段 `_type:'List'` 兼容补丁。原因:数据集 parquet 的 HF 元数据里 `state`/`actions` 是旧版 `_type:'List'`,`datasets==3.6.0` 把 `List` 从类型注册表删了,LeRobot 走 `load_dataset("parquet")` 加载会报 `ValueError: Feature type 'List' not found`。补丁在运行时把 `List` 注册成 `Sequence`(定长 7 的语义等价物),**不改数据文件**(同事 data_loader 同款做法,已端到端验证)。放 `ur5_policy.py` 顶部是因为 config.py 会 import 它(早于数据加载);单独跑 `LeRobotDataset(...)` 时需先 `import openpi.policies.ur5_policy`。

```python
import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# --- datasets `_type:'List'` 兼容补丁 ----------------------------------------
# datasets>=3.0 删了 `List` 类型(只留 LargeList/Sequence),加载旧 parquet 报
# `ValueError: Feature type 'List' not found`。运行时把 List 注册成 Sequence
# (Sequence 支持 feature+length,定长 7 的语义等价物),不改数据文件。
# 必须在任何 LeRobotDataset 加载前执行;本模块被 config.py import,早于数据加载。
import datasets.features.features as _dff

if "List" not in _dff._FEATURE_TYPES:
    _dff._FEATURE_TYPES["List"] = _dff.Sequence
# ---------------------------------------------------------------------------


def make_ur5_example() -> dict:
    """随机输入样例,测试 policy server / 走查 transform 用。"""
    return {
        "state": np.random.rand(7),
        "image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "Pick up the yellow ping-pong ball and place it in the white box.",
    }


def _parse_image(image) -> np.ndarray:
    # 与 libero_policy.py 同款:LeRobot 训练时存 float32 (C,H,W),推理时客户端发 uint8 (H,W,C)。
    # 统一成 uint8 (H,W,C)。
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class UR5Inputs(transforms.DataTransformFn):
    """把 UR5 数据集观测映射成模型输入。

    数据集 `state` 已是拼接好的 7 维(joint_0..5 + gripper),图像在 `image`/`wrist_image`。
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["image"])
        wrist_image = _parse_image(data["wrist_image"])
        inputs = {
            "state": np.asarray(data["state"]),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),   # 无右腕,填黑图
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # PI05 用 False(只有 PI0_FAST 才 True)
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class UR5Outputs(transforms.DataTransformFn):
    """取前 7 维动作(6 关节 + 夹爪)。`...` 兼容带/不带 batch 维。"""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :7])}
```

> 这是 [libero_policy.py](../src/openpi/policies/libero_policy.py) 的逐行对照版,只改了 key 名(`observation/image`→`image` 等)和 `state` 直接取(不再拼接)。`_parse_image` 用仓库同款实现,不要自己造。

**② 数据工厂** —— 加进 [config.py](../src/openpi/training/config.py)(或 `training/misc/`):

```python
@dataclasses.dataclass(frozen=True)
class LeRobotUR5DataConfig(DataConfigFactory):
    """UR5 ping-pong 数据集。用【绝对关节位置动作】,配套数据集自带的绝对 norm_stats。"""

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # 数据集 key 已是 state/image/wrist_image/actions/prompt,与 UR5Inputs 读取的一致。
        # RepackTransform 用同各映射,顺带过滤掉 frame_index/timestamp 等无关 key。
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "state": "state",
                        "image": "image",
                        "wrist_image": "wrist_image",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[ur5_policy.UR5Inputs(model_type=model_config.model_type)],
            outputs=[ur5_policy.UR5Outputs()],
        )
        # 不加 DeltaActions:数据集 actions 是绝对关节位置,自带的 norm_stats 也是绝对的,
        # 两者配套。若改用增量动作,必须同时重算 norm_stats(见 D.2)。
        model_transforms = ModelTransformFactory()(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
```

> 注意:这里**没有** `DeltaActions`(README 模板有,但我们不用)。原因见 D.1 —— 数据集自带 norm_stats 是绝对动作的统计量。`ur5_policy` 需在 config.py 顶部 `import openpi.policies.ur5_policy as ur5_policy`(仿 `import ... libero_policy`)。

**③ 注册 TrainConfig** —— 加进 [config.py::_CONFIGS](../src/openpi/training/config.py) 列表:

```python
TrainConfig(
    name="pi05_ur_ping_pong",
    # pi05 + LoRA:LoRA 靠 variant 字符串触发(不是 lora= 字段),配 freeze_filter + ema_decay=None。
    # action_dim 保持默认 32(模型内部动作维度),7 维 state/action 由 PadStatesAndActions 补零、
    # UR5Outputs 取前 7 —— 和 pi05_libero(7 维)同款,保证和 pi05_base 权重形状一致。别设 action_dim=7。
    model=pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ),
    data=LeRobotUR5DataConfig(
        repo_id="local/ur_pick_up_ping_pong",                    # 见 C.2:HF_LEROBOT_HOME 下
        assets=AssetsConfig(asset_id="ur5e"),                    # 见 D.2:复用数据集自带 norm_stats
        base_config=DataConfig(prompt_from_task=True),           # prompt 从 task 字段取
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"        # π0.5 微调起点;自动保留 .*lora.* 新参数
    ),
    freeze_filter=pi0_config.Pi0Config(                          # 与 model 完全一致的构造,再取 freeze filter
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ).get_freeze_filter(),
    ema_decay=None,                                              # LoRA 不用 EMA
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=1_000, peak_lr=5e-5, decay_steps=20_000, decay_lr=5e-6,
    ),
    num_train_steps=20_000,     # 130 ep 小数据,别训太多步,防过拟合
    batch_size=32,              # 8 卡数据并行 → 每卡 4;OOM 就调小,必须能被 8 整除
),
```

> 关键点(都已核对源码):
> - **LoRA 写法**:`paligemma_variant="gemma_2b_lora"` + `action_expert_variant="gemma_300m_lora"`(rank/alpha 在 [gemma.py](../src/openpi/models/gemma.py) 里自动定:2b→rank16,300m→rank32)+ `freeze_filter=...get_freeze_filter()` + `ema_decay=None`。参考 `pi0_libero_low_mem_finetune`。仓库**没有** pi05+LoRA 现成 config,这是自己组合的(`get_freeze_filter` 对 pi05 同样适用)。
> - **action_dim 不设**:保持 32,和 base 权重对齐;真实 7 维靠 pad/截取。
> - **CheckpointWeightLoader** 的 `_merge_params` 用 `missing_regex=".*lora.*"`,加载 base 时**保留新初始化的 LoRA 权重**(base 里没 LoRA 层是正常的)。
>
> 字段含义、默认值、何时改:见 [train_custom_pi0.md 第七节](train_custom_pi0.md#七字段调整总览)。

### C.2 本地数据集怎么被 `repo_id` 找到

LeRobot 库找数据集的顺序:`repo_id` → 先查 `HF_LEROBOT_HOME/<repo_id>`,再查 HuggingFace Hub。你的数据在 `/home/jh/OpenPI_UR/datasets/ur_pick_up_ping_pong/`,不在默认位置。两种做法:

- **设环境变量**:`export HF_LEROBOT_HOME=/home/jh/OpenPI_UR/datasets`,然后 `repo_id="ur_pick_up_ping_pong"`(无 `local/` 前缀)。
- **或用 root 参数**:`LeRobotDataset(repo_id, root=...)` —— 但 config 里 `repo_id` 是字符串,走不到 root。**推荐用 HF_LEROBOT_HOME 环境变量**。远程机器把数据集 rsync 过去后同样设这个变量。

> 训练前先在远程验证能加载:`uv run python -c "from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; ds=LeRobotDataset('ur_pick_up_ping_pong'); print(len(ds), ds[0].keys())"`(已 `export HF_LEROBOT_HOME=...`)。

### C.3 LoRA 微调的开关(JAX)

LoRA 不是顶层 CLI flag,而是在 model config 里选带 `lora` 的变体 + 设 `freeze_filter`(参考 `pi0_libero_low_mem_finetune` 之类的写法):

- `paligemma_variant="gemma_2b_lora"`、`action_expert_variant="gemma_300m_lora"`
- `freeze_filter=model.get_freeze_filter()` —— 自动冻结主体、只训 LoRA 参数
- [CheckpointWeightLoader](../src/openpi/training/weight_loaders.py) 的 `_merge_params` 用 `missing_regex=".*lora.*"`,会**保留新初始化的 LoRA 权重**(不从 base 加载),所以 base 里没有 LoRA 层是正常的。

> PyTorch 后端**不支持 LoRA**。要 LoRA 必须走 JAX。

> **检查点 C**:`uv run python -c "from openpi.training.config import get_config; print(get_config('pi05_ur_ping_pong'))"` 能打印出 config,且 `data.create(...)` 不报错(在远程跑,需先 sync 代码 + 设 HF_LEROBOT_HOME)。

---

## D. 计算 norm_stats

### D.1 两种 norm_stats,别混淆

- **lerobot 数据集自带的 stats**(`meta/episodes_stats.jsonl`):每模态 min/max/mean/std,lerobot 库自己用,不是 openpi 的。
- **openpi 的 NormStats**:mean/std/**q01/q99**,PI0.5 用 **quantile 归一化**(`use_quantile_norm=True`,由 `model_type != PI0` 自动开启,见 [data_loader.py](../src/openpi/training/data_loader.py) 的 `create_base_config`)。你本地 `datasets/ur_pick_up_ping_pong/norm_stats.json` 里有 q01/q99 → **这是 openpi 格式**,应是同事之前算的。

### D.2 本配置的做法:复用数据集自带的绝对 norm_stats(不重算)

数据集根目录的 `norm_stats.json` 就是 openpi 格式(state+actions 各 mean/std/**q01/q99**,7 维),且是**绝对动作**的统计(actions 均值≈state 均值≈1.43,不是 ≈0)。本配置用绝对动作(无 DeltaActions),所以**直接复用它**,不用重算、也不借 base 的 ur5e stats(base 那套是增量的,会和绝对动作错配)。

config 里 `assets=AssetsConfig(asset_id="ur5e")`(`assets_dir=None`)→ [create_base_config](../src/openpi/training/config.py) 从本地 `assets_dirs/asset_id` 加载,即 `./assets/pi05_ur_ping_pong/ur5e/norm_stats.json`(`assets_dirs = assets_base_dir/name`,`assets_base_dir` 默认 `./assets`)。所以训练前把数据集的 norm_stats 摆到位:

```bash
# 远程,在 repo 目录
source $OPENPI_HOME/env.sh
cd $OPENPI_HOME/repo
mkdir -p assets/pi05_ur_ping_pong/ur5e
cp $HF_LEROBOT_HOME/ur_pick_up_ping_pong/norm_stats.json assets/pi05_ur_ping_pong/ur5e/norm_stats.json
ls -l assets/pi05_ur_ping_pong/ur5e/norm_stats.json    # 确认 ~1.9K
```

### D.3 何时要重算(备选)

- **改用增量动作**(加 `DeltaActions(make_bool_mask(6,-1))`):必须重算,因为自带的是绝对统计。去掉 `assets=AssetsConfig(...)` 或保留 `asset_id="ur5e"` 但清空上面那个文件,跑:

```bash
cd $OPENPI_HOME/repo
uv run scripts/compute_norm_stats.py pi05_ur_ping_pong --max_frames 5000
# 写到 ./assets/pi05_ur_ping_pong/<asset_id>/norm_stats.json,与上面的加载路径一致
```

[compute_norm_stats.py](../scripts/compute_norm_stats.py) `main(config_name, max_frames=None)`:positional 传 config 名,只对 `["state","actions"]` 算统计量,跑的是 config 的 data_transforms(含/不含 DeltaActions 自动一致)。

> **检查点 D**:`ls assets/pi05_ur_ping_pong/ur5e/norm_stats.json` 存在且 ~1.9K(复用),或 compute 后存在(重算)。`use_quantile_norm` 对 pi05 自动 True,所以必须有 q01/q99 —— 自带的就有。

---

## E. 远程训练(JAX + LoRA)

### E.1 准备(远程,每次开终端先 source)

```bash
source $OPENPI_HOME/env.sh          # 设 HF_LEROBOT_HOME / OPENPI_DATA_HOME / UV_CACHE_DIR,并 cd repo
# 代码已改好:C 节新增的 ur5_policy.py(含 _type:'List' 补丁)+ config.py(注册 pi05_ur_ping_pong)
# norm_stats 已就位(D.2):assets/pi05_ur_ping_pong/ur5e/norm_stats.json
# 数据集已解压:HF_LEROBOT_HOME/ur_pick_up_ping_pong/  ← 原始 tgz 直接解压,List bug 由 ur5_policy.py 补丁自动处理,无需跑 fix 脚本

# 冒烟测试:config 能解析 + 数据能加载
# 第一行 import config 会顺带 import ur5_policy → 触发 _type:'List' 补丁(C.1①)。
# 第二行单独跑 LeRobotDataset,必须显式先 import ur5_policy,否则补丁不触发、会撞 List bug。
uv run python -c "from openpi.training.config import get_config; c=get_config('pi05_ur_ping_pong'); print(c.name, c.model.pi05, c.batch_size)"
uv run python -c "import openpi.policies.ur5_policy; from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; ds=LeRobotDataset('ur_pick_up_ping_pong'); print(len(ds), list(ds[0].keys()))"
```

> `uv run` 必须在 `$OPENPI_HOME/repo` 下(`env.sh` 已 cd 过去),否则回落到 conda base 无 jax。

### E.2 开训(8× RTX 5090,JAX 数据并行)

```bash
cd $OPENPI_HOME/repo
WANDB_DISABLED=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_ur_ping_pong \
    --exp_name=ur_0721_lora \
    --overwrite \
    --checkpoint_base_dir $OPENPI_HOME/checkpoints \
    --log_interval 100 \
    --save_interval 1000 \
    --keep_period 5000
```

- JAX 自动用全部 8 张卡做数据并行,`batch_size=32` → 每卡 4。**batch_size 必须能被 8 整除**。
- **OOM 才加 FSDP 分片**:`--fsdp_devices 8`(把模型分片到 8 卡,省显存但略慢)。LoRA 本身省显存,一般不用。
- `WANDB_DISABLED=true` 跳过 wandb 登录(共享账号别乱绑);想用 wandb 就去掉这个并 `wandb login`。
- `--checkpoint_base_dir` 把 ckpt 写到大盘 `$OPENPI_HOME/checkpoints`(默认 `./checkpoints` 在 repo 内也行,reacher-fast 139G 够)。

关键 CLI([train.py](../scripts/train.py),tyro 把每个 TrainConfig 字段变成 `--field` flag):

| flag | 作用 |
|---|---|
| 第一个位置参数 | config 名(`pi05_ur_ping_pong`) |
| `--exp_name` | **必填**,实验名;checkpoint 存到 `<checkpoint_base_dir>/<config>/<exp_name>/<step>/` |
| `--overwrite` | 目录已存在则清空(与 `--resume` 互斥) |
| `--resume` | 从最新 ckpt 续训 |
| `--num_train_steps` `--batch_size` `--save_interval` `--keep_period` | 覆盖 config 默认 |
| `--fsdp_devices` | FSDP 分片设备数(默认 1=纯数据并行) |
| `--checkpoint_base_dir` `--assets_base_dir` | 改 ckpt / assets 根目录 |
| `--wandb_enabled` | 默认 True;关掉用 `--no-wandb_enabled` 或 `WANDB_DISABLED=true` |

> PyTorch 全参微调本路线用不到(JAX LoRA 已确认可行)。若将来要 PyTorch:先 `uv run examples/convert_jax_model_to_pytorch.py` 把 pi05_base 转成 `model.safetensors`,config 设 `pytorch_weight_path`,再 `uv run torchrun --standalone --nproc_per_node=8 scripts/train_pytorch.py pi05_ur_ping_pong --exp_name=...`。

### E.3 训练时看什么

- **loss 曲线**:wandb(若开)或终端日志。flow matching loss 应平稳下降。**loss 不降**优先查:动作绝对/增量是否搞反、norm_stats 是否对上、state/actions 维度是否错位。
- **过拟合信号**:130 ep 小数据,20k 步可能就过拟合。看验证(若切了 val)或早停,多留几个 ckpt(`--save_interval 1000 --keep_period 1000`)便于挑。
- **OOM**:调小 `--batch_size`;JAX 用 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` 放开显存上限。

> **检查点 E**:`checkpoints/pi05_ur_ping_pong/ur_0720_lora/<step>/` 下有 `params/`(JAX)或 `model.safetensors`(PyTorch)+ `assets/ur5e/`(或对应 asset_id 的 norm_stats)+ `train_state/`。能进 F 节。

---

## F. 仿真评估

> ⚠️ 重申:**openpi 没有 UR 仿真**。本节给三条路,按你的目的选。

### F.1 路 A(推荐先用):借 libero 仿真验证"流程闭环"

目的不是评估 UR 策略好坏,而是**验证"训好的 checkpoint → serve_policy → 客户端 rollout"整条链没断**。用 [examples/libero/](../examples/libero/) 的现成 sim:

```bash
# 远程:起一个 libero 策略服务(用官方 pi05_libero ckpt,不是你的 UR ckpt)
uv run scripts/serve_policy.py policy:default --env=libero --port 8000

# 另一终端:跑 libero rollout
cd examples/libero
uv run main.py --env libero_spatial ...   # 见 examples/libero/README.md
```

走通后,说明你的 serve/客户端链路 OK;UR 侧只要把"客户端发的 obs dict"换成 UR 格式即可(见 G)。

### F.2 路 B:自建 UR5e 仿真(PyBullet / MuJoCo)

真正评估 UR 策略需要自己搭 sim。最小可行做法:

1. **仿真环境**:PyBullet + UR5e URDF(或 MuJoCo + UR5e MJCF)。负责:正向/反向运动学、夹爪、乒乓球+白盒的碰撞与渲染。
2. **相机渲染**:base 相机 + wrist 相机,出 [224,224,3] uint8 图(与训练数据视角一致)。
3. **控制频率**:与训练数据 fps 对齐(见 B.3-4,先问清同事)。
4. **rollout 脚本**:仿照 [examples/libero/main.py](../examples/libero/main.py) 写,用 `WebsocketClientPolicy` 连服务端:

```python
from openpi_client import WebsocketClientPolicy
from openpi_client.runtime import ActionChunkBroker

policy = WebsocketClientPolicy(host="GPU机器IP", port=8000)
broker = ActionChunkBroker(policy)   # 动作分块下发,减少推理频率
obs = sim.reset()
for step in range(max_steps):
    action_chunk = broker.infer(obs_to_dict(obs))   # 发 UR 格式 obs
    for a in action_chunk:                          # 或按 broker 逻辑
        obs = sim.step(a)
        if done: break
```

5. **指标**:任务成功率(乒乓球进白盒)、完成时间、动作平滑度。多 seed 多初始位姿统计。

### F.3 路 C:Isaac Sim / IsaacLab

若团队有 Isaac Sim,可用 IsaacLab 的 UR5e 任务。代价是环境重、安装复杂,但渲染质量高、物理可信。接入点同 F.2 第 4 步(rollout 脚本不变,只换 sim 后端)。

> **关于 sim-to-real gap**:sim 里渲染的图像与真机相机差异大时,策略会失效。UR 数据是真机采集的,在 sim 里评估本质是"域外测试"。**仿真评估通过 ≠ 真机能跑通**,但能验证策略没出"乱动/卡死/维度错"这类低级问题。真机表现以 G 节为准。

> **检查点 F**:至少在某个 sim(或 libero)里看到策略输出的 action chunk 数值范围合理(state 量级内、无 NaN、夹爪开合有变化)、机械臂动作方向大致符合任务。

---

## G. 真机部署 UR5e

架构(见 [架构总览](architecture.md)):GPU 机器跑 `serve_policy.py`(重),真机旁的机器跑你写的客户端(轻,只装 `openpi-client`,不装 JAX),websocket 通信。

### G.1 服务端(GPU 机器)

```bash
cd ~/projects/openpi
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_ur_ping_pong \
    --policy.dir=checkpoints/pi05_ur_ping_pong/ur_0720_lora/20000 \
    --port 8000
# 后台常驻 + 日志
nohup uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_ur_ping_pong \
    --policy.dir=checkpoints/pi05_ur_ping_pong/ur_0720_lora/20000 \
    --port 8000 > serve.log 2>&1 &  echo $! > serve.pid
```

- 全局参数(`--port`/`--record`/`--default-prompt`)**必须在 `policy:checkpoint` 子命令前面**([train_custom_pi0.md 第 8.3 节](train_custom_pi0.md#83--参数顺序规则易错))。
- `--policy.config` 决定用哪套 UR 变换;`--policy.dir` 指向训好的 ckpt(含权重 + norm_stats)。两者**必须配套**。
- UR 没有 `policy:default --env=ur5`,必须走 `policy:checkpoint`。

### G.2 真机客户端(自己写)

模板:[examples/simple_client/main.py](../examples/simple_client/main.py)、[examples/libero/main.py](../examples/libero/main.py)。核心循环:

```python
import numpy as np
from openpi_client import WebsocketClientPolicy
from openpi_client.runtime import ActionChunkBroker
# 你自己装的 UR 接口,例如 ur_rtde
import rtde_control, rtde_receive

HOST, PORT = "GPU机器IP", 8000
policy = WebsocketClientPolicy(host=HOST, port=PORT)
broker = ActionChunkBroker(policy)

rtde_c = rtde_control.RTDEControlInterface("192.168.1.10")   # UR 控制器 IP
rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.10")
# 相机:base 相机 + wrist 相机,出 uint8 (H,W,3) 224x224

def get_obs():
    q = np.array(rtde_r.getActualQ())            # 6 关节
    gripper = np.array([read_gripper()])         # 夹爪状态,归一化到训练时的范围
    return {
        "state": np.concatenate([q, gripper]).astype(np.float32),  # [7],与训练 state 同布局
        "image": base_camera.read(),             # [224,224,3] uint8
        "wrist_image": wrist_camera.read(),       # [224,224,3] uint8
        "prompt": "Pick up the yellow ping-pong ball and place it in the white box.",
    }

# 安全:先servoJ 慢速试一段,确认动作方向
while not done:
    obs = get_obs()
    actions = broker.infer(obs)          # [action_horizon, 7],绝对关节位置(已在服务端 AbsoluteActions 还原)
    for a in actions:
        rtde_c.servoJ(a[:6].tolist(), vel, acc, dt, lookahead, gain)
        set_gripper(a[6])
        # dt 与训练 fps 对齐(见 B.3-4)
```

关键点:

- **obs 格式必须和 `UR5Inputs` 期望的一致**:`state`[7]、`image`/`wrist_image` key 名、uint8 (H,W,3)。key 名错了或图像通道顺序错了,策略直接废。先用 `--record` 录一段,拿 [examples/policy_records.ipynb](../examples/policy_records.ipynb) 回放核对 obs 是否和训练数据同分布。
- **`ActionChunkBroker`**:推理一次拿一整段 action horizon,分多步下发,减少推理频率开销。理解它的"何时重新推理"逻辑(见 [action_chunk_broker.py](../packages/openpi-client/src/openpi_client/action_chunk_broker.py))。
- **夹爪范围/关节单位**:训练数据的 gripper 是 0.37–1.0(从 norm_stats 看)、关节是弧度。真机读取/下发要换算到**同一套单位与范围**,否则策略输出对不上真机。
- **控制频率**:servoJ 的 `dt` 要匹配训练 fps(B.3-4),不匹配会导致动作播放过快/过慢。
- **安全**:UR 真机先低速、限位、急停在手;第一次跑用 `--record`、人在旁边、随时 e-stop。

### G.3 客户端依赖(真机旁机器,轻量)

```bash
# 不需要装 JAX/大模型,只装 openpi-client
pip install openpi-client ur_rtde  # 或按 examples/simple_client/requirements.txt
```

> **检查点 G**:服务端 `serve.log` 无报错、客户端 `policy.get_server_metadata()` 能拿到 metadata、首帧推理返回 shape 正确的 action chunk、机械臂在低速下动作方向大致符合任务。

---

## H. 排错速查

| 现象 | 先查 |
|---|---|
| 训练 loss 不降 | 动作绝对/增量是否搞反;norm_stats 是否对上;state/actions 维度错位;图像通道顺序 |
| `get_config` 找不到 name | config 是否注册进 `_CONFIGS`;是否 sync 到远程 |
| 数据集加载报错 | `HF_LEROBOT_HOME` 是否设对;`repo_id` 是否匹配目录名 |
| JAX 看不到 GPU | CUDA/cuDNN 版本;`jax.devices()` 输出;是否装了 jax-cuda |
| 推理 OOM | `XLA_PYTHON_CLIENT_MEM_FRACTION`;改小 batch;确认没和同事的 serve 抢卡 |
| 真机动作乱跳 | obs key 名/图像通道/单位/范围与训练不一致;用 `--record` 核对 |
| 真机动作速度不对 | servoJ `dt` 与训练 fps 不匹配(B.3-4) |
| 推理后端报错 | checkpoint 目录有无 `model.safetensors` 决定 JAX/PyTorch,需与训练后端一致 |

---

## I. 需要你向同事确认的事(建议先问清)

1. 远程机器地址、GPU 型号/显存、openpi 目录路径、是否已 `uv sync`、是否已下载 base 权重到 `~/.cache/openpi`。
2. 他们用的 `pi05_ur_ping_pong` config 完整定义(repo_id、action_horizon、是否 LoRA、assets 用 base 还是自算、batch_size、num_train_steps)。**有现成 diff 直接要**。
3. 数据采集频率(10Hz 还是 20Hz 降采样)、真机打算用多少 Hz 下发。
4. `ur_0706_130p_lora` 那次训练的效果/问题(便于对比)。
5. 真机 UR 具体型号(确认 UR5e?)、夹爪型号(Robotiq 2F-85?)、相机型号与安装位置、UR 控制器 IP 与接口(RTDE?)。
6. 是否有已搭好的 UR 仿真(F 节)。

把这些问清,本文 C/G 节的"按你实际情况调整"处就能定死。

---

## J. 下一步深入

- 配置/变换/服务理论:[train_custom_pi0.md](train_custom_pi0.md)
- 数据流串讲:[data_flow.md](data_flow.md)
- 逐文件目录:[architecture.md](architecture.md)
- 客户端闭环:[openpi-client/](../packages/openpi-client/src/openpi_client/)(`WebsocketClientPolicy`+`ActionChunkBroker`+`runtime`)
- UR5 接入模板:[examples/ur5/README.md](../examples/ur5/README.md)
- 归一化统计:[norm_stats.md](../docs/norm_stats.md)
