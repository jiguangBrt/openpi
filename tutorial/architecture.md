# openPI 架构详解

本文件系统性地介绍 [openpi](https://github.com/Physical-Intelligence/openpi) 仓库的整体架构、每个目录以及其中的关键文件作用，帮助读者快速建立对整个项目的认知。

openpi 由 Physical Intelligence 团队开源，包含三类机器人视觉-语言-动作（VLA）模型：

- **π₀**：基于 flow matching 的 VLA 模型。
- **π₀-FAST**：基于 FAST 动作 tokenizer 的自回归 VLA 模型。
- **π₀.₅**：π₀ 的升级版，使用 knowledge insulation 训练，具备更好的开放世界泛化能力。

仓库同时支持 **JAX** 与 **PyTorch** 两套实现（训练与推理）。

---

## 目录结构总览

```
openpi/
├── src/openpi/            # 核心库：模型、策略、训练、数据变换、服务
│   ├── models/            # JAX 模型实现（π₀ / π₀-FAST / Gemma / SigLIP / ViT / LoRA）
│   ├── models_pytorch/    # PyTorch 版模型实现
│   ├── policies/          # 策略封装（推理时把观测→动作）
│   ├── training/          # 训练配置、数据加载、checkpoint、优化器、分片
│   ├── serving/           # websocket 推理服务端
│   ├── shared/            # 通用工具（下载、图像、归一化、类型、nnx 工具）
│   └── transforms.py      # 数据变换框架（输入/输出归一化、tokenizer 等）
├── packages/openpi-client/# 独立客户端库（机器人侧调用）
├── scripts/               # 命令行入口：train / serve_policy / compute_norm_stats
├── examples/              # 各机器人平台示例（ALOHA / DROID / LIBERO / UR5 …）
├── third_party/           # 第三方硬件/数据依赖（aloha, libero）
├── docs/                  # 额外文档（docker、远程推理、norm_stats）
├── pyproject.toml         # 包定义与依赖（使用 uv 管理）
└── README.md              # 项目主文档
```

下面逐层展开。

---

## 1. `src/openpi/` —— 核心库

整个项目的核心代码都在这里，按职责拆分为若干子模块。

### 1.1 顶层文件

| 文件 | 作用 |
|------|------|
| `transforms.py` | **数据变换框架**。定义 `DataTransformFn` 协议与一系列变换：`Normalize` / `Unnormalize`（状态与动作归一化）、`ResizeImages`、`DeltaActions` / `AbsoluteActions`（动作表示）、`TokenizePrompt`、`TokenizeFASTInputs` / `ExtractFASTActions`（FAST 模型专用）、`PadStatesAndActions`、`RepackTransform`、`InjectDefaultPrompt` 等。`Group` 把 input/output 变换成对管理，`compose` 串起多个变换。这是把“各平台原始数据”映射到“模型统一输入格式”的核心。 |
| `transforms_test.py` | 上述变换的单元测试。 |
| `conftest.py` | pytest 公共 fixture。 |
| `py.typed` | 标记本包提供类型注解（PEP 561）。 |

### 1.2 `models/` —— JAX 模型实现

JAX/Flax 实现的全部模型组件。

| 文件 | 作用 |
|------|------|
| `model.py` | **模型抽象与数据格式**。定义 `ModelType` 枚举（PI0 / PI0_FAST / PI05）、`Observation` / `Actions` 数据类、`BaseModelConfig`（`create`/`load`/`load_pytorch`）、`BaseModel`（`sample_actions`）、`restore_params`（从 checkpoint 恢复参数）。规定模型期望的图像键 `IMAGE_KEYS` 与分辨率 `(224,224)`，并描述了统一的嵌套字典输入格式。 |
| `pi0.py` | **π₀ 模型实现**。flow matching 头，包含注意力掩码构造 `make_attn_mask`、位置编码 `posemb_sincos` 等，组装 PaliGemma 视觉-语言骨干 + 动作专家。 |
| `pi0_fast.py` | **π₀-FAST 模型实现**。自回归动作生成，使用 FAST tokenizer，含 `left_to_right_align` 等专门处理。 |
| `pi0_config.py` | `Pi0Config` / `Pi05Config` 等 `BaseModelConfig` 子类，定义模型超参（层数、维度、动作 horizon 等）。 |
| `gemma.py` | Gemma 语言模型骨干（源自 big_vision），π₀ 的 LLM 部分。 |
| `gemma_fast.py` | π₀-FAST 专用的 Gemma 变体。 |
| `siglip.py` | SigLIP 视觉编码器（图像→token），源自 big_vision。 |
| `vit.py` | ViT 视觉骨干实现。 |
| `tokenizer.py` | 文本 tokenizer（基于 sentencepiece / HF Processor）与权重加载。 |
| `lora.py` | LoRA 低秩适配器实现，用于低成本微调。 |
| `lora_test.py` / `model_test.py` / `pi0_test.py` / `tokenizer_test.py` | 各组件测试。 |
| `utils/fsq_tokenizer.py` | FSQ（Finite Scalar Quantization）tokenizer，FAST 动作离散化用。 |

### 1.3 `models_pytorch/` —— PyTorch 模型实现

与 `models/` 对应的 PyTorch 版本（2025 年 9 月新增支持）。

| 文件 | 作用 |
|------|------|
| `pi0_pytorch.py` | π₀ 的 PyTorch 实现，`PaliGemmaWithExpertModel` 的动作专家与采样逻辑。 |
| `gemma_pytorch.py` | Gemma + PaliGemma 的 PyTorch 实现。 |
| `preprocessing_pytorch.py` | PyTorch 下的观测预处理（图像归一化、位置编码等）。 |
| `transformers_replace/` | 对 HuggingFace transformers 部分模块的替换/补丁，以适配 PaliGemma expert 架构。 |

> 是否走 PyTorch 由 checkpoint 目录下是否存在 `model.safetensors` 自动判定（见 `policy_config.py`）。

### 1.4 `policies/` —— 策略封装

把“训练好的模型 + 变换”封装成可推理的 `Policy`，并为各机器人平台提供输入/输出变换。

| 文件 | 作用 |
|------|------|
| `policy.py` | **核心 `Policy` 类**。继承自客户端的 `BasePolicy`，持有模型与 input/output transforms，`infer()` 流程：应用输入变换 → `model.sample_actions` → 应用输出变换。同时支持 JAX 与 PyTorch 模型。 |
| `policy_config.py` | `create_trained_policy()`：根据 `TrainConfig` + checkpoint 构建可推理策略，自动检测 JAX/PyTorch、加载 norm_stats、装配变换链。 |
| `aloha_policy.py` | ALOHA 平台专用变换 `AlohaInputs`/`AlohaOutputs`，处理 4 相机、14 维状态、关节/夹爪空间转换（`adapt_to_pi`）。 |
| `droid_policy.py` | DROID 平台变换 `DroidInputs`/`DroidOutputs`，处理外置/腕部相机、关节+夹爪状态。 |
| `libero_policy.py` | LIBERO 仿真基准变换。 |
| `policy_test.py` | 策略测试。 |

### 1.5 `training/` —— 训练管线

| 文件 | 作用 |
|------|------|
| `config.py` | **训练配置中枢**。定义 `TrainConfig`（含 model / data / optimizer / weight_loader / checkpoint 等）、`DataConfig`、`AssetsConfig`，以及各类 `DataConfigFactory`（`LeRobotAlohaDataConfig`、`LeRobotLiberoDataConfig`、`RLDSDroidDataConfig`、`LeRobotDROIDDataConfig`、`SimpleDataConfig`、`FakeDataConfig`）。底部 `_CONFIGS` 列表登记所有可用配置名（如 `pi0_aloha_sim`、`pi05_droid`），`get_config(name)` 按名查找。 |
| `data_loader.py` | 数据加载器。定义 `Dataset` 协议（随机访问），基于 LeRobot dataset 构造训练样本，应用 transforms，多进程预取。 |
| `droid_rlds_dataset.py` | DROID 专用的 RLDS 格式数据加载器（LeRobot 对超大数据集不够可扩展时使用），含 DROID 专用过滤/变换。 |
| `checkpoints.py` | 基于 Orbax 的 checkpoint 管理：`initialize_checkpoint_dir`、保存/恢复、keep_period、resume。 |
| `weight_loaders.py` | `WeightLoader` 协议及实现：从 base checkpoint 加载/转换权重（如 base→微调、JAX↔PyTorch 适配）。 |
| `optimizer.py` | 优化器与学习率调度：`CosineDecaySchedule` 等基于 optax。 |
| `sharding.py` | JAX FSDP 分片策略，`make_mesh` 构造 batch/fsdp 网格。 |
| `utils.py` | 训练通用工具函数。 |
| `misc/` | 额外平台配置：`polaris_config.py`（PolaRiS 基线）、`roboarena_config.py`（RoboArena）。 |

### 1.6 `serving/` —— 推理服务

| 文件 | 作用 |
|------|------|
| `websocket_policy_server.py` | `WebsocketPolicyServer`：把一个 `Policy` 通过 websocket 暴露为服务，支持 `load`/`infer`，用 msgpack 序列化 numpy 数组。与客户端 `WebsocketClientPolicy` 对接。 |

### 1.7 `shared/` —— 通用工具

| 文件 | 作用 |
|------|------|
| `download.py` | 从 `gs://openpi-assets` 下载 checkpoint/资产并缓存到 `~/.cache/openpi`（可用 `OPENPI_DATA_HOME` 覆盖）。 |
| `normalize.py` | `NormStats`（mean/std/q01/q99）、`RunningStats`（在线统计），用于状态/动作归一化统计量计算与存储。 |
| `image_tools.py` | 图像格式转换、resize、uint8/float 处理。 |
| `array_typing.py` | 基于 `jaxtyping` 的数组类型注解（`at.Float`、`at.Real`、`at.KeyArrayLike` 等）。 |
| `nnx_utils.py` | Flax NNX 相关辅助（变量管理、pytree 操作）。 |
| `__init__.py` / 各 `*_test.py` | 包初始化与测试。 |

---

## 2. `packages/openpi-client/` —— 独立客户端库

机器人侧使用的轻量客户端，**独立于主库**，避免在机器人上安装 JAX/大依赖。

```
packages/openpi-client/src/openpi_client/
├── base_policy.py            # BasePolicy 抽象（infer/reset）
├── websocket_client_policy.py# WebsocketClientPolicy：通过 websocket 调用远端服务
├── action_chunk_broker.py    # ActionChunkBroker：把动作分块逐条下发
├── msgpack_numpy.py          # numpy 数组的 msgpack 序列化
├── image_tools.py            # 客户端侧图像工具
└── runtime/                  # 运行时框架（环境-智能体-订阅者编排）
    ├── runtime.py            # Runtime：编排 environment/agent/subscribers 的主循环
    ├── environment.py        # Environment 抽象（机器人/仿真环境）
    ├── agent.py              # Agent 抽象（观测→动作）
    ├── subscriber.py         # Subscriber 抽象（日志/录制/回调）
    └── agents/               # 具体智能体实现（如 policy_agent）
```

典型用法：机器人程序用 `WebsocketClientPolicy` 连接 `serve_policy.py` 起的服务，可选套一层 `ActionChunkBroker` 控制动作分块频率。

---

## 3. `scripts/` —— 命令行入口

| 文件 | 作用 |
|------|------|
| `train.py` | **JAX 训练入口**。解析配置名，初始化 wandb、数据加载器、优化器、FSDP 分片，训练循环，checkpoint 保存。 |
| `train_pytorch.py` | **PyTorch 训练入口**，对应 `models_pytorch`。 |
| `serve_policy.py` | **推理服务入口**。`tyro` 解析参数，按配置名/checkpoint 构建 `Policy`，用 `WebsocketPolicyServer` 暴露。支持 `Default`（环境默认策略）或 `Checkpoint`（指定配置+目录）。 |
| `compute_norm_stats.py` | 计算并保存归一化统计量（`NormStats`），微调前对自有数据集运行。 |
| `train_test.py` | 训练冒烟测试。 |
| `docker/` | Docker 部署：`compose.yml`、`serve_policy.Dockerfile`、安装脚本。 |

---

## 4. `examples/` —— 机器人平台示例

每个子目录是一个完整示例，演示如何在某平台上采集数据、转换、训练、部署。

| 目录 | 内容 |
|------|------|
| `aloha_sim/` | ALOHA 仿真（Gym） rollout 示例。`env.py`（环境）、`main.py`（用 client runtime 跑策略）、`saver.py`（录制）。 |
| `aloha_real/` | 真实 ALOHA 机器人部署。`real_env.py`、`robot_utils.py`、`convert_aloha_data_to_lerobot.py`（数据转 LeRobot 格式）。 |
| `droid/` | DROID（Franka）示例。`main.py`（部署）、`convert_droid_data_to_lerobot.py`、`compute_droid_nonidle_ranges.py`（空闲帧过滤）、`README_train.md`（全量训练指南）。 |
| `libero/` | LIBERO 仿真基准。`convert_libero_data_to_lerobot.py`、`main.py`。 |
| `simple_client/` | 最小客户端示例，连接已部署的策略做推理。 |
| `ur5/` | UR5 微调教程（README 形式，示范如何定义 `UR5Inputs`/`UR5Outputs` 变换）。 |
| `inference.ipynb` | 推理 notebook 教程。 |
| `policy_records.ipynb` | 回放录制数据的 notebook。 |
| `convert_jax_model_to_pytorch.py` | JAX checkpoint → PyTorch 转换脚本。 |

---

## 5. `third_party/` —— 第三方依赖

| 目录 | 作用 |
|------|------|
| `aloha/` | ALOHA 硬件相关（ROS 包、CMake、STL 模型、gripper 装配），仅在真实 ALOHA 部署时需要。 |
| `libero/` | LIBERO 基准代码（作为子模块）。 |

---

## 6. `docs/` —— 补充文档

| 文件 | 内容 |
|------|------|
| `docker.md` | Docker 安装与使用指南。 |
| `remote_inference.md` | 远程推理（策略服务跑在强 GPU 上，机器人侧通过 websocket 调用）。 |
| `norm_stats.md` | 归一化统计量的作用、计算与重载（微调时复用 base 模型统计量）。 |

---

## 7. 顶层配置文件

| 文件 | 作用 |
|------|------|
| `pyproject.toml` | 包定义、依赖、工具配置（ruff 等），用 `uv` 管理环境。 |
| `uv.lock` | uv 依赖锁文件。 |
| `.python-version` | Python 版本固定。 |
| `.pre-commit-config.yaml` | 预提交钩子。 |
| `.gitmodules` | Git 子模块（LeRobot、libero 等）。 |
| `CONTRIBUTING.md` | 贡献指南。 |
| `LICENSE` / `LICENSE_GEMMA.txt` | 许可证（Gemma 权重有单独许可）。 |

---

## 8. 端到端数据流（串联各模块）

理解架构最好的方式是看数据如何流过系统。

### 训练流
```
配置名 (e.g. "pi05_droid")
  → training/config.py::get_config()        取出 TrainConfig
  → config.DataConfigFactory.create()        构造 DataConfig（LeRobot/RLDS…）
  → training/data_loader.py                  加载样本，套上 transforms.Group(inputs)
  → transforms.py                            归一化、动作表示、tokenize、pad
  → models/model.py (Observation/Actions)    转成模型输入格式
  → models/pi0.py 或 models_pytorch/         前向 + loss（flow matching / 自回归）
  → training/optimizer.py + sharding.py      optax 优化 + FSDP 分片
  → training/checkpoints.py                  Orbax 保存 checkpoint + assets(norm_stats)
```

### 推理流
```
serve_policy.py  --config <name> --dir <ckpt>
  → policy_config.create_trained_policy()    自动检测 JAX/PyTorch，加载权重 + norm_stats
  → policies/policy.py::Policy               持有 model + transforms
  → serving/websocket_policy_server.py       暴露 websocket 服务

机器人侧:
  openpi_client.WebsocketClientPolicy        连接服务
  → (可选) ActionChunkBroker                 动作分块
  → client.runtime.Runtime                   编排 environment/agent 循环
  → Policy.infer(obs)                        服务端：input transform → sample_actions → output transform → 返回动作
```

### 数据变换流（`transforms.py` 是枢纽）
```
平台原始观测 (各平台字段名各异)
  → *_policy.py 中的 Inputs 变换             统一到模型期望的 image/state/actions/prompt
  → transforms.Normalize                      用 NormStats 归一化
  → transforms.TokenizePrompt / FAST 专用     文本/动作离散化
  → transforms.PadStatesAndActions            对齐维度
  → 模型
  → (输出) transforms.Unnormalize / ExtractFASTActions → 平台动作空间
```

---

## 9. 关键概念速查

- **配置驱动**：一切以 `training/config.py` 中的 `_CONFIGS` 为中心，通过配置名（`pi0_aloha_sim`、`pi05_droid`…）选择模型、数据、优化器、权重加载方式。
- **变换解耦**：`transforms.py` + 各 `*_policy.py` 把“平台异构数据”与“模型统一格式”解耦，新平台只需写一组 Inputs/Outputs 变换（参考 `examples/ur5/README.md`）。
- **双后端**：JAX（`models/`，主路径）与 PyTorch（`models_pytorch/`）共用配置与变换，由 checkpoint 文件自动选择。
- **客户端/服务端分离**：重依赖（JAX/大模型）只在服务端，机器人侧用轻量 `openpi-client`，便于远程推理与避免依赖冲突。
- **资产与 norm_stats**：归一化统计量作为 asset 随 checkpoint 保存，可复用 base 模型的统计量（见 `AssetsConfig` 与 `docs/norm_stats.md`）。

---

## 10. 推荐阅读顺序

1. `README.md` —— 安装、可用模型与 checkpoint。
2. `src/openpi/training/config.py` —— 浏览 `_CONFIGS`，理解配置如何组装。
3. `src/openpi/transforms.py` + `src/openpi/policies/aloha_policy.py` —— 理解数据如何映射到模型。
4. `src/openpi/models/model.py` —— 理解模型抽象与统一数据格式。
5. `scripts/train.py` 与 `scripts/serve_policy.py` —— 看训练与推理入口。
6. `examples/aloha_sim/` —— 跑通一个最小端到端示例。
7. `docs/` 三篇文档 —— 补充 docker、远程推理、归一化细节。
