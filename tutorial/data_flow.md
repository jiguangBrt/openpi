# openpi 数据流串讲

> 本文用**一条数据流**把 openpi 的核心骨架串起来。读完你会理解:模型只认一种统一格式,不同机器人平台的异构数据靠一条"变换链"翻译成这种格式。
>
> 配套阅读:[architecture.md](architecture.md)(逐文件目录详解)。
>
> 所有源码引用均可点击跳转(相对路径基于本 `tutorial/` 目录,指向 `../`)。

## 核心思想:配置驱动 + 变换解耦

> **模型只认一种统一格式;不同机器人平台的异构数据,靠一条"变换链"翻译成这种格式。**

所以只要搞懂三件事:

1. **统一格式长什么样** — [../src/openpi/models/model.py](../src/openpi/models/model.py)
2. **变换链怎么组装** — [../src/openpi/transforms.py](../src/openpi/transforms.py) + [../src/openpi/training/config.py](../src/openpi/training/config.py)
3. **变换链怎么接到模型上跑** — [../src/openpi/policies/policy.py](../src/openpi/policies/policy.py) + [../src/openpi/policies/policy_config.py](../src/openpi/policies/policy_config.py) + [../scripts/serve_policy.py](../scripts/serve_policy.py)

---

## 一、统一格式:模型只认这个

文件:[../src/openpi/models/model.py](../src/openpi/models/model.py)

模型不关心你是 ALOHA 还是 UR,它只认一个嵌套字典:

```python
{
  "image": {
      "base_0_rgb":       [h, w, 3],   # 基座/外部相机
      "left_wrist_0_rgb": [h, w, 3],   # 左腕相机
      "right_wrist_0_rgb":[h, w, 3],   # 右腕相机
  },
  "image_mask": { ... bool ... },      # 哪些相机有效(缺失的填黑图 + mask=False)
  "state":  float32 [s],               # 低维机器人状态(关节角等)
  "tokenized_prompt": int32 [l],       # 文字指令的 token
  "tokenized_prompt_mask": bool [l],
  "actions": float32 [ah, ad],         # 动作序列(action_horizon × action_dim)
}
```

关键约定(见 [model.py:39-47](../src/openpi/models/model.py)):

- `IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")` — 模型固定要这三个视角的名字。
- `IMAGE_RESOLUTION = (224, 224)`。
- 图像范围 `[-1, 1]` float32(uint8 输入会在 `Observation.from_dict` 自动转换)。

这个字典被 [`Observation.from_dict`](../src/openpi/models/model.py) 包成结构化的 `Observation` 对象喂给模型。**模型侧的 `sample_actions` 只吃 `Observation`,吐 `actions`。**

> 不同平台的差异,全部要在"变成上面这个字典"之前解决掉 — 这就是变换链的职责。

---

## 二、变换链:把任意平台数据翻译成统一格式

文件:[../src/openpi/transforms.py](../src/openpi/transforms.py) — 整个项目的**枢纽**。

它定义了一个协议 [`DataTransformFn`](../src/openpi/transforms.py):

```python
class DataTransformFn(Protocol):
    def __call__(self, data: dict) -> dict: ...   # 进一个 dict,出一个 dict
```

每个变换就是"吃 dict 吐 dict"的函数。openpi 提供一抽屉现成积木:

| 变换 | 作用 |
|---|---|
| [`RepackTransform`](../src/openpi/transforms.py) | 改 key 名/重组结构(把平台字段映射到统一字段) |
| [`InjectDefaultPrompt`](../src/openpi/transforms.py) | 没给指令时注入默认 prompt |
| [`ResizeImages`](../src/openpi/transforms.py) | 图片 resize 到 224×224 |
| [`Normalize`](../src/openpi/transforms.py) / [`Unnormalize`](../src/openpi/transforms.py) | 用 norm_stats 做状态/动作归一化(z-score 或 quantile) |
| [`DeltaActions`](../src/openpi/transforms.py) / [`AbsoluteActions`](../src/openpi/transforms.py) | 绝对动作 ↔ 相对(增量)动作转换 |
| [`TokenizePrompt`](../src/openpi/transforms.py) | 文字指令 → token |
| [`TokenizeFASTInputs`](../src/openpi/transforms.py) / [`ExtractFASTActions`](../src/openpi/transforms.py) | FAST 模型专用的动作离散化 |
| [`PadStatesAndActions`](../src/openpi/transforms.py) | 把 state/actions 零填充到模型的 action_dim |

变换按"输入方向"和"输出方向"成对管理,容器是 [`Group`](../src/openpi/transforms.py):

```python
@dataclasses.dataclass(frozen=True)
class Group:
    inputs:  Sequence[DataTransformFn] = ()   # 喂给模型前正向应用
    outputs: Sequence[DataTransformFn] = ()   # 模型输出后反向应用
```

[`compose`](../src/openpi/transforms.py) 把一串变换串成管道([`CompositeTransform`](../src/openpi/transforms.py),逐个 apply)。[`Group.push`](../src/openpi/transforms.py) 往链上追加变换(**注意**:`inputs` 追加到末尾,`outputs` 追加到**开头** — 因为输出方向是逆序执行的,这样才能和输入方向形成正确的逆变换对)。

---

## 三、变换链怎么组装:配置驱动

文件:[../src/openpi/training/config.py](../src/openpi/training/config.py) — 理解整个项目的**钥匙**。

所有可用任务都在文件末尾的 [`_CONFIGS`](../src/openpi/training/config.py) 列表里(约 560 行起),每个是一个 [`TrainConfig`](../src/openpi/training/config.py):

```python
TrainConfig(
    name="pi0_aloha_sim",                      # 配置名(唯一标识)
    model=pi0_config.Pi0Config(),              # 用哪个模型(π0/π0.5/π0-FAST + 超参)
    data=LeRobotAlohaDataConfig(...),          # 数据从哪来 + 用哪些变换
    weight_loader=CheckpointWeightLoader(...), # 从哪个 base checkpoint 加载权重
    num_train_steps=20_000,                    # 训练超参
    ...
)
```

[`get_config(name)`](../src/openpi/training/config.py) 按 name 查字典。**远程那个 `pi05_ur_ping_pong` 就是同事往这个 `_CONFIGS` 里加的一条。**

### 三层变换的装配(关键)

[`DataConfig`](../src/openpi/training/config.py) 把变换分成**三层**,按顺序应用:

```
原始数据
  │
  ├─ repack_transforms   只在训练时用:把 LeRobot 数据集的字段名重映射到推理时的字段名
  │                       (因为数据集存的 key 和机器人在线发的 key 可能不一样)
  ├─ data_transforms     平台专属变换:ALOHA的AlohaInputs、DROID的DroidInputs...
  │                       + DeltaActions(把绝对动作转增量)
  ├─ Normalize            用 norm_stats 归一化(由 create_base_config 自动加)
  └─ model_transforms     模型专属变换:TokenizePrompt + ResizeImages + PadStatesAndActions
                                                  (由 ModelTransformFactory 按 model_type 生成)
                                                          ↓
                                                     喂给模型
```

看 [`ModelTransformFactory`](../src/openpi/training/config.py):它根据 `model_type`(PI0/PI05/PI0_FAST)返回不同的 model_transforms。比如 PI0/PI05 就是 `[InjectDefaultPrompt, ResizeImages(224,224), TokenizePrompt(...), PadStatesAndActions(action_dim)]`。

而 `data_transforms` 由各 [`DataConfigFactory`](../src/openpi/training/config.py) 子类填:

- [`LeRobotAlohaDataConfig`](../src/openpi/training/config.py) → 装上 `AlohaInputs`/`AlohaOutputs` + 可选 `DeltaActions`
- [`LeRobotLiberoDataConfig`](../src/openpi/training/config.py) → 装上 `LiberoInputs`/`LiberoOutputs`
- [`RLDSDroidDataConfig`](../src/openpi/training/config.py) / [`LeRobotDROIDDataConfig`](../src/openpi/training/config.py) → 装上 `DroidInputs`/`DroidOutputs`
- [`SimpleDataConfig`](../src/openpi/training/config.py) → 通用,自己传变换
- [`FakeDataConfig`](../src/openpi/training/config.py) → 调试用假数据

**这就是"新平台接入"的全部套路**:写一个 `<Platform>Inputs`/`<Platform>Outputs`(放进 [../src/openpi/policies/](../src/openpi/policies/)),再写一个 `DataConfigFactory` 子类把它们装进 `data_transforms`,然后在 `_CONFIGS` 里注册一个 `TrainConfig`。[../examples/ur5/README.md](../examples/ur5/README.md) 就是这个流程的教程 — 同事的 UR 乒乓球任务就是照这个改的。

---

## 四、平台变换长什么样(ALOHA 为例)

文件:[../src/openpi/policies/aloha_policy.py](../src/openpi/policies/aloha_policy.py)

[`AlohaInputs`](../src/openpi/policies/aloha_policy.py) 做的事就是:**把 ALOHA 平台的观测,翻译成模型要的统一字典**。

输入(ALOHA 原始):

```python
{"state": [14], "images": {"cam_high":..., "cam_left_wrist":...}, "prompt": "..."}
```

注意 ALOHA 图片是 `[channel, height, width]`,状态是 `[左臂6关节, 左夹爪, 右臂6关节, 右夹爪]`。

`AlohaInputs.__call__` 做三件事:

1. [`_decode_aloha`](../src/openpi/policies/aloha_policy.py):图片 `c h w → h w c`;状态做 `adapt_to_pi` 转换(关节翻转、夹爪从线性空间转回角度空间 — 因为 π0 base 模型是在角度空间训的)。
2. 把相机名映射到模型要的 `base_0_rgb` / `left_wrist_0_rgb` / `right_wrist_0_rgb`,**缺失的相机填黑图 + mask=False**(这就是 `image_mask` 的由来)。
3. 输出统一字典:`{"image":..., "image_mask":..., "state":..., "actions":...(训练时才有), "prompt":...}`。

[`AlohaOutputs`](../src/openpi/policies/aloha_policy.py) 是反方向:模型输出 `actions` 后,截取前 14 维,做 `adapt_to_pi` 的逆变换,还给 ALOHA 的动作空间。

> **输入变换 = 平台→统一;输出变换 = 统一→平台。** 成对出现,互为逆操作。这就是为什么 `Group` 有 `inputs` 和 `outputs` 两列。

其他平台同理:[droid_policy.py](../src/openpi/policies/droid_policy.py)、[libero_policy.py](../src/openpi/policies/libero_policy.py)。

---

## 五、推理时怎么串起来

文件:[../src/openpi/policies/policy_config.py](../src/openpi/policies/policy_config.py) + [../src/openpi/policies/policy.py](../src/openpi/policies/policy.py)

[`create_trained_policy`](../src/openpi/policies/policy_config.py) 是**推理装配的总装车间**。给定一个 `TrainConfig` + checkpoint 目录,它:

1. **自动判断后端**([policy_config.py:48-50](../src/openpi/policies/policy_config.py)):checkpoint 目录里有 `model.safetensors` → PyTorch;否则 → JAX。
2. **加载模型**:JAX 走 [`restore_params`](../src/openpi/models/model.py) + `model.load`;PyTorch 走 `load_pytorch`。
3. **加载 norm_stats**:从 checkpoint 的 `assets/` 目录读(确保推理用的归一化统计量和训练时一致)。
4. **拼装变换链**([policy_config.py:75-89](../src/openpi/policies/policy_config.py)),最精华的部分 — 把三层变换拼成一条完整的 input 链和一条 output 链:

```
input_transforms = [
    *repack_transforms.inputs,        # (推理时通常为空)
    InjectDefaultPrompt(default_prompt),
    *data_transforms.inputs,          # AlohaInputs 等
    Normalize(norm_stats),            # 归一化
    *model_transforms.inputs,         # TokenizePrompt + ResizeImages + Pad
]
output_transforms = [
    *model_transforms.outputs,        # (FAST 的 ExtractFASTActions)
    Unnormalize(norm_stats),          # 反归一化
    *data_transforms.outputs,         # AlohaOutputs 等
    *repack_transforms.outputs,
]
```

注意顺序:input 是 `平台 → 归一化 → 模型格式`;output 是 `模型格式 → 反归一化 → 平台`。**完全对称的逆过程。**

5. 把 model + 两条链塞进 [`Policy`](../src/openpi/policies/policy.py)。

### `Policy.infer` 的执行

见 [policy.py:67](../src/openpi/policies/policy.py):

```python
def infer(self, obs: dict) -> dict:
    inputs = self._input_transform(obs)              # 跑完整 input 变换链
    inputs = 加 batch 维 + 转 jax.Array / torch.Tensor
    observation = Observation.from_dict(inputs)      # dict → 结构化 Observation
    actions = self._sample_actions(rng, observation) # 模型推理
    outputs = {"state":..., "actions": actions}
    outputs = 去 batch 维 → numpy
    outputs = self._output_transform(outputs)        # 跑完整 output 变换链
    return outputs
```

**就这么简单:输入变换 → 模型 → 输出变换。** JAX 和 PyTorch 走同一个 `infer`,只是张量转换分支不同([policy.py:72-79](../src/openpi/policies/policy.py))。

---

## 六、服务入口

文件:[../scripts/serve_policy.py](../scripts/serve_policy.py) — 远程正在跑的脚本(原版)。

它用 `tyro` 解析参数,核心两种模式:

- `policy:checkpoint --policy.config=<name> --policy.dir=<ckpt>`:从指定 config + checkpoint 构建 Policy。
- `policy:default --env=aloha_sim`:用 [`DEFAULT_CHECKPOINT`](../scripts/serve_policy.py) 表里预置的 config/checkpoint。

[`main`](../scripts/serve_policy.py) 就三步:[`create_policy`](../scripts/serve_policy.py) → (可选)套 [`PolicyRecorder`](../src/openpi/policies/policy.py) 录制 → 起 [`WebsocketPolicyServer`](../src/openpi/serving/websocket_policy_server.py) 在 `0.0.0.0:8000` 监听。

机器人侧用 `openpi-client` 的 `WebsocketClientPolicy` 连这个端口,发观测、收动作(msgpack 序列化 numpy)。**这就是 client/server 分离:重依赖只在服务端,机器人侧轻量。** 详见 [../docs/remote_inference.md](../docs/remote_inference.md)。

---

## 七、端到端数据流总图

```
┌─────────────── 训练 ────────────────┐
│ config名 → get_config() → TrainConfig │
│   → data.create() 装配三层 transforms  │
│   → data_loader 取 LeRobot 样本        │
│   → repack → data(平台) → Normalize → model 变换
│   → Observation → model.compute_loss   │
│   → optax + FSDP → Orbax 保存 ckpt+assets(norm_stats)
└──────────────────────────────────────┘

┌─────────────── 推理 ──────────────────────────────────┐
│ serve_policy.py --config <name> --dir <ckpt>           │
│   → create_trained_policy()                             │
│       ├─ 检测 model.safetensors → JAX or PyTorch        │
│       ├─ 加载权重 + norm_stats                          │
│       └─ 装配 input/output 变换链 → Policy              │
│   → WebsocketPolicyServer 监听 :8000                    │
│                                                        │
│   机器人侧: WebsocketClientPolicy → (ActionChunkBroker)
│        ↓ 收到 obs                                       │
│   Policy.infer(obs):                                    │
│     input_transform: 平台obs→归一化→模型格式            │
│     Observation.from_dict → model.sample_actions        │
│     output_transform: 模型格式→反归一化→平台动作         │
│        ↓ 返回 actions                                   │
└────────────────────────────────────────────────────────┘
```

训练相关文件:[../scripts/train.py](../scripts/train.py)、[../src/openpi/training/data_loader.py](../src/openpi/training/data_loader.py)、[../src/openpi/training/checkpoints.py](../src/openpi/training/checkpoints.py)、[../src/openpi/training/optimizer.py](../src/openpi/training/optimizer.py)、[../src/openpi/training/sharding.py](../src/openpi/training/sharding.py)、[../src/openpi/training/weight_loaders.py](../src/openpi/training/weight_loaders.py)。

---

## 八、回头看远程那个进程

现在能解读远程跑的命令了:

```
python scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_ur_ping_pong \
  --policy.dir=/data/001_models/pi05_ur_ping_pong/ur_0706_130p_lora/59999/
```

- `policy:checkpoint` 模式 → 走 [`create_trained_policy`](../src/openpi/policies/policy_config.py)。
- `config=pi05_ur_ping_pong` → 同事在 [`config.py::_CONFIGS`](../src/openpi/training/config.py) 加的自定义配置,model 是 `Pi0Config(pi05=True)`,data 应该是一个 UR 专用的 `DataConfigFactory`(带 `UR...Inputs`/`UR...Outputs`)。
- `dir=.../59999/` → LoRA 微调到 59999 步的 checkpoint。`ur_0706_130p` 大概是 "7月6日、130段数据" 的 LoRA 实验。
- 这个目录里**没有** `model.safetensors`(因为它是 JAX LoRA),所以走 JAX 后端,占 18GB 显存加载 π0.5 权重。

同事改 [config.py](../src/openpi/training/config.py)(加 `pi05_ur_ping_pong`)、[serve_policy.py](../scripts/serve_policy.py)(可能改默认端口/逻辑)、[data_loader.py](../src/openpi/training/data_loader.py)(可能加 UR 数据加载)的 diff,等原版吃透后一看就懂。

---

## 九、下一步

- **深入模型内部**:[../src/openpi/models/pi0.py](../src/openpi/models/pi0.py)(flow matching 怎么采样动作)、[../src/openpi/models/pi0_config.py](../src/openpi/models/pi0_config.py)(模型超参)、[../src/openpi/models/gemma.py](../src/openpi/models/gemma.py) + [../src/openpi/models/siglip.py](../src/openpi/models/siglip.py)(PaliGemma 视觉语言骨干)、[../src/openpi/models/lora.py](../src/openpi/models/lora.py)(LoRA 适配器)。
- **看训练循环**:[../scripts/train.py](../scripts/train.py) + [../src/openpi/training/data_loader.py](../src/openpi/training/data_loader.py) + [../src/openpi/training/checkpoints.py](../src/openpi/training/checkpoints.py)。
- **norm_stats 细节**:[../src/openpi/shared/normalize.py](../src/openpi/shared/normalize.py) + [../docs/norm_stats.md](../docs/norm_stats.md)。
- **客户端**:[../packages/openpi-client/src/openpi_client/](../packages/openpi-client/src/openpi_client/)。
