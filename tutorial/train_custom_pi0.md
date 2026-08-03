# 在自己的数据/机械臂上定制 π0

> 本文汇总"为新平台接入、微调、部署 π0/π0.5"所需的核心知识:配置体系、三层变换装配、新平台接入套路、字段调整总览、推理服务启动。
>
> 配套阅读:[architecture.md](architecture.md)(逐文件目录详解)、[data_flow.md](data_flow.md)(一条数据流串讲)。
>
> 所有源码引用均可点击跳转(相对路径基于本 `tutorial/` 目录,指向 `../`)。

---

## 一、配置驱动:一切从 `TrainConfig` 开始

openpi 是**配置驱动**的。每个可用任务都是 [config.py::_CONFIGS](../src/openpi/training/config.py) 末尾列表里的一条 [`TrainConfig`](../src/openpi/training/config.py),[`get_config(name)`](../src/openpi/training/config.py) 按名查找。

一个微调 config 的骨架(以 `pi0_aloha_pen_uncap` 为例):

```python
TrainConfig(
    name="pi0_aloha_pen_uncap",                       # ① 唯一配置名
    model=pi0_config.Pi0Config(),                     # ② 用哪个模型架构
    data=LeRobotAlohaDataConfig(                      # ③ 数据从哪来 + 变换串(工厂)
        repo_id="physical-intelligence/aloha_pen_uncap_diverse",
        assets=AssetsConfig(assets_dir="gs://...pi0_base/assets", asset_id="trossen"),
        default_prompt="uncap the pen",
        repack_transforms=...,
    ),
    weight_loader=CheckpointWeightLoader("gs://...pi0_base/params"),  # ④ 训练起点权重
    num_train_steps=20_000,                           # ⑤ 训练步数
)
```

`TrainConfig` 绝大多数字段有默认值,真正常填的就 **name + model + data + weight_loader + num_train_steps** 五样(详见第七节字段总览)。

> ⚠️ `weight_loader`(训练起点)和推理时的 `--policy.dir`(训好的 checkpoint)是两码事:训练用 `weight_loader` 从 base 接着训,推理用 `--policy.dir` 加载训完的权重。推理走 `create_trained_policy` 时不读 `weight_loader`。

---

## 二、三层变换装配(核心)

### 模型只认统一格式

模型([model.py](../src/openpi/models/model.py))不关心你是 ALOHA 还是 UR,只认一个嵌套字典:

```python
{
  "image":        {"base_0_rgb":[h,w,3], "left_wrist_0_rgb":..., "right_wrist_0_rgb":...},
  "image_mask":   {...bool...},      # 哪些相机有效
  "state":        float32 [action_dim],
  "tokenized_prompt": int32 [l],
  "tokenized_prompt_mask": bool [l],
  "actions":      float32 [action_horizon, action_dim],
}
```

约定:`IMAGE_KEYS=("base_0_rgb","left_wrist_0_rgb","right_wrist_0_rgb")`,`IMAGE_RESOLUTION=(224,224)`,图像 `[-1,1]` float32。**不同平台的差异,全部在"变成这个字典"之前解决 —— 这就是变换链的职责。**

### 三层的定义、填充、装配

**定义**:[`DataConfig`](../src/openpi/training/config.py) 用三个 `Group` 字段表达三层:

```python
class DataConfig:
    repack_transforms: Group = Group()   # ①
    data_transforms:   Group = Group()   # ②
    model_transforms:  Group = Group()   # ③
    norm_stats: ...
    use_quantile_norm: bool = False
```

**填充**:由 [`DataConfigFactory`](../src/openpi/training/config.py) 子类的 `create()` 方法在运行时填入。`create()` 内部把平台 `Inputs/Outputs`、`DeltaActions`、`ModelTransformFactory` 的产物拼进三个 `Group`。

**装配**:有两个装配点,把 `*inputs` 展开、中间手动插 `Normalize`:

- 训练:[data_loader.py:185-190](../src/openpi/training/data_loader.py)
- 推理:[policy_config.py:77-89](../src/openpi/policies/policy_config.py)

```
input_transforms = [
    *repack_transforms.inputs,        # ① (推理时通常空 Group)
    InjectDefaultPrompt(default_prompt),
    *data_transforms.inputs,          # ② (XxxInputs, DeltaActions)
    Normalize(norm_stats),            #    归一化(手动插入,不算独立一层)
    *model_transforms.inputs,         # ③ (InjectDefaultPrompt, ResizeImages, TokenizePrompt, Pad)
]
output_transforms = [                 # 完全对称的逆过程
    *model_transforms.outputs,        # ③'(FAST 才有 ExtractFASTActions)
    Unnormalize(norm_stats),          #    反归一化
    *data_transforms.outputs,         # ②'(AbsoluteActions, XxxOutputs)
    *repack_transforms.outputs,       # ①'
]
```

> 关键:`Normalize`/`Unnormalize` **不属于三层字段**,是装配时手动插在 ② 和 ③ 之间的公共步骤。`Group.push` 的"inputs 追尾、outputs 追头"规则,保证 `DeltaActions`/`AbsoluteActions` 成对逆变换在正反链位置对应。

### 每层各做什么

| 层 | 做什么 | 产出 |
|---|---|---|
| ① repack | **只改 key 名**(数据集存储 key → 平台 Inputs 认的标准 key),值不动 | key 拉齐后的 dict |
| ② data | **平台语义→统一语义**:关节/夹爪空间转换、相机名映射、缺失相机填黑图+mask、拼 state、DeltaActions | 统一格式雏形(未归一化、uint8 图) |
| — Normalize | state/actions 做 z-score(PI0)或 quantile(PI0.5/FAST) | 归一化后量级 |
| ③ model | **按 model_type 把数据变成模型张量**:InjectDefaultPrompt、ResizeImages、TokenizePrompt、PadStatesAndActions | 可喂模型的 dict |

`ModelTransformFactory` 按 `model_type` 分支([config.py:107-163](../src/openpi/training/config.py)):PI0/PI05 用 `[InjectDefaultPrompt, ResizeImages, TokenizePrompt, PadStatesAndActions]`,FAST 用 `TokenizeFASTInputs`+`ExtractFASTActions`。

---

## 三、新平台接入套路

接入一个新机械臂只需三步(详见 [examples/ur5/README.md](../examples/ur5/README.md)):

### 1. 写一组平台变换 `XxxInputs` / `XxxOutputs`(放进 [`policies/`](../src/openpi/policies/))

`XxxInputs`(`DataTransformFn`)做三件事:拼 `state`、相机名映射到 `IMAGE_KEYS`(缺失的填黑图+`image_mask=False`)、透传 `actions`/`prompt`。`XxxOutputs` 是逆操作(截到机械臂真实动作维度)。

### 2. 写一个 `DataConfigFactory` 子类,重写 `create()`(放进 `training/config.py` 或 `training/misc/`)

```python
@dataclasses.dataclass(frozen=True)
class LeRobotXxxDataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs, model_config) -> DataConfig:
        repack_transform = Group(inputs=[RepackTransform({...})])        # ①
        data_transforms = Group(inputs=[XxxInputs(...)], outputs=[XxxOutputs()])
        data_transforms = data_transforms.push(                          # 可选 Δ
            inputs=[DeltaActions(make_bool_mask(...))],
            outputs=[AbsoluteActions(make_bool_mask(...))],
        )
        model_transforms = ModelTransformFactory()(model_config)         # ③ 照搬
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),          # 自动填 norm_stats
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
```

`create_base_config` 自动加载 norm_stats、按 `model_type` 设 `use_quantile_norm`,不用管。

### 3. 在 `_CONFIGS` 注册一条 `TrainConfig`

```python
TrainConfig(name="pi0_xxx", model=pi0_config.Pi0Config(),
            data=LeRobotXxxDataConfig(repo_id="your_user/xxx_dataset", ...),
            weight_loader=CheckpointWeightLoader("gs://...pi0_base/params"),
            num_train_steps=30_000)
```

### 4.(数据准备)转换 + 算 norm_stats

- 数据转 LeRobot 格式(`examples/*/convert_*_data_to_lerobot.py`)
- 微调前跑 `scripts/compute_norm_stats.py --config-name=pi0_xxx`

> 关键认知:**变换串不是在 config 里手写的,而是 `data=<工厂>(几个旋钮)` 在运行时由 `create()` 拼出来的。** 第③层(`ModelTransformFactory`)和 `create_base_config` 都是现成的,新平台完全不用碰模型侧。

---

## 四、具体例子:UR 机械臂数据流

UR5:6 DoF + 1 夹爪 = 7 维动作;外部相机(base)+ 腕部相机(wrist),无右腕。配置 `pi0_ur5`(PI0,`action_dim` 默认 32,`action_horizon=50`)。

**起点**(LeRobot 样本):
```python
{"image": float32[3,224,224], "wrist_image": float32[3,224,224],
 "joints": float32[6], "gripper": float32[1], "actions": float32[50,7],
 "prompt": "pick up the ping pong ball"}
```

**① repack** —— 改 key 名(`base_rgb←image`, `wrist_rgb←wrist_image`),值不动。

**② `UR5Inputs`**:
- `state = concat(joints[6], gripper[1])` → `[7]`
- 图片 `_parse_image`:LeRobot `float32 (C,H,W)` → `uint8 (H,W,C)`
- 映射到 `base_0_rgb`/`left_wrist_0_rgb`,右腕填黑图+`mask=False`
```python
{"state":[7], "image":{base_0_rgb, left_wrist_0_rgb(真), right_wrist_0_rgb(黑)},
 "image_mask":{..., right_wrist_0_rgb:False}, "actions":[50,7], "prompt":"..."}
```

**② `DeltaActions(mask=make_bool_mask(6,-1))`** —— 前6维关节转增量(`actions[...,:6] -= state[...,:6]`),夹爪维保持绝对。

**Normalize** —— state/actions 做 z-score。

**③ `model_transforms`**:
- `ResizeImages(224,224)`
- `TokenizePrompt`:`prompt` → `tokenized_prompt` int32[l] + mask
- `PadStatesAndActions(action_dim=32)`:state→[32], actions→[50,32]

**终点**:
```python
{"image":{3 个 uint8[224,224,3]}, "image_mask":{...},
 "state": float32[32], "tokenized_prompt": int32[l], "tokenized_prompt_mask": bool[l],
 "actions": float32[50,32]}
```

**逆向(推理输出链)**:`Unnormalize` → `AbsoluteActions`(Δ 加回 state)→ `UR5Outputs`(`actions[:,:7]` 截 7 维)→ 返回 `[50,7]` 绝对动作给 `ActionChunkBroker` 分块下发。

---

## 五、几个关键概念

### 5.1 mask

项目里有多种 mask,别混淆:

| mask | 在哪产生 | 含义 |
|---|---|---|
| `image_mask` | ② 层 `Inputs` | 每个相机视角真/假(黑图占位)。`False` 的 token 在注意力里被屏蔽,见 [pi0.py:113-123](../src/openpi/models/pi0.py) |
| `tokenized_prompt_mask` | ③ 层 `TokenizePrompt` | 文字 token 序列里哪些是真 token、哪些是 padding |
| DeltaActions `make_bool_mask` | ② 层装配时 | 动作各维"要不要转增量"的掩码,如 `make_bool_mask(6,-1)` = 前6维转增量、夹爪维保持绝对 |
| `token_ar_mask`/`token_loss_mask` | FAST 专用 ③ 层 | 自回归顺序掩码 / 哪些 token 算 loss |

> repack **不做 mask**,只改名;`image_mask` 在 ② 层产生,`tokenized_prompt_mask` 在 ③ 层产生。

### 5.2 policy ≠ model

**策略(policy)π(a|o) 是"观测→动作的完整映射"**,不是模型本身。openpi 的 [`Policy`](../src/openpi/policies/policy.py) = **模型 + 输入变换链 + 输出变换链**。光有模型没用 —— 没变换链,模型吃不到平台观测、吐不出平台动作。

所以推理命令的 `policy.*` 参数和 `policy:` 子命令,命名都指向"构建一个 Policy"这个整体,而非"加载模型"。两个参数是 policy 的两个半身:

| | `policy.config`(配方) | `policy.dir`(权重) |
|---|---|---|
| 本质 | 代码里写好的结构定义 | 训练产出的参数文件 |
| 内容 | 模型架构 + 变换链 + 归一化方式 | 权重 + norm_stats |
| 决定了 | policy "长什么样" | policy "会什么" |

两者必须配套:config 搭骨架,dir 填血肉,合体才是能跑的 policy。

### 5.3 工厂旋钮机制

`data=LeRobotAlohaDataConfig(...)` 传入的是**工厂 + 几个旋钮**,不是现成变换串。真正的三层 `Group` 在 `create()` 运行时拼出。"制定变换串"= 选工厂子类(决定用哪套平台 Inputs)→ 拧旋钮(`default_prompt`/`use_delta_joint_actions`/`adapt_to_pi`/`extra_delta_transform`/`repack_transforms`)→ `create()` 按旋钮拼装。

### 5.4 config 长短差异的根源

**config 长度 = 你偏离工厂默认值的程度。** 工厂把"标准情况"做成默认值/内部硬编码 → 数据集标准就短(Libero);数据集非标准(相机数不同、key 名不同、固定 prompt、复用 base 统计量)→ 要显式覆盖,config 就长(pen_uncap 三相机+固定 prompt+借 trossen 统计量)。

### 5.5 `assets`/`base_config` 为何"有时写有时不写"

`assets`、`base_config`、`repo_id` 都从基类 `DataConfigFactory` 继承([config.py:172-179](../src/openpi/training/config.py)),都有默认回退:

| 字段 | 默认 | 写的情况 | 不写的情况 |
|---|---|---|---|
| `repo_id` | `MISSING` | 真训练几乎必填 | 只有 FakeDataConfig 调试 |
| `assets` | 空(用 repo_id 当 asset_id,读自己数据集的 norm_stats) | 小数据集借 base 统计量 | 数据集大、自己算过 norm_stats |
| `base_config` | `DataConfig()`(`prompt_from_task=False`) | 多任务要动态 prompt | 单任务用 `default_prompt` 代替 |

回退链([create_base_config](../src/openpi/training/config.py)):`asset_id = self.assets.asset_id or repo_id`;`assets_dir = self.assets.assets_dir or assets_dirs`。

---

## 六、怎么知道该拧哪个旋钮

从**数据集事实**出发逐层对照:

| 数据集的事实 | 拧哪个旋钮 | 怎么拧 |
|---|---|---|
| key 名/相机数和默认不同 | `repack_transforms` | 重写 `RepackTransform`,把数据集 key 映射到 Inputs 认的标准 key |
| 动作是绝对值 | `use_delta_joint_actions`/`extra_delta_transform` | 开(True),夹爪维 mask=False |
| 动作已是增量 | 同上 | 关(False) |
| 单任务固定指令 | `default_prompt` | 设字符串 |
| 多任务每回合不同 | `base_config=DataConfig(prompt_from_task=True)` | 从 task 字段取 |
| 数据集小,复用 base 统计量 | `assets=AssetsConfig(assets_dir=base, asset_id=某机器人)` | 指向 base 的 assets |
| 数据集大,用自己统计量 | 不写 `assets`,先跑 `compute_norm_stats.py` | 默认用 repo_id 当 asset_id |

> 最稳的实操:**在 `_CONFIGS` 里抄一个最像的 config 当模板,只改和数据集对不上的几处。** 不确定默认值时看工厂类源码或 `--help`。

---

## 七、字段调整总览

### 7.1 `TrainConfig` 顶层([config.py:471](../src/openpi/training/config.py))

| 字段 | 默认 | 什么时候改 |
|---|---|---|
| `name` | 必填 | 总要,唯一配置名 |
| `model` | `Pi0Config()` | 换模型:`Pi0Config(pi05=True)` 等 |
| `data` | `FakeDataConfig` | 总要,选 `DataConfigFactory` 子类 |
| `weight_loader` | `NoOpWeightLoader` | **微调必改**,指向 base 权重 |
| `num_train_steps` | `30000` | 按数据量调 |
| `batch_size` | `32` | 显存不够时调小 |
| `num_workers` | `2` | IO 慢时调大 |
| `optimizer`/`lr_schedule` | `AdamW`/`CosineDecay` | 一般不动 |
| `freeze_filter` | `nnx.Nothing` | **LoRA 微调时**用 `model.get_freeze_filter()` |
| `save_interval`/`keep_period` | `1000`/`5000` | 想多留 ckpt |

### 7.2 `DataConfigFactory` 通用旋钮(基类)

| 字段 | 默认 | 什么时候改 |
|---|---|---|
| `repo_id` | `MISSING` | 真训练必填 |
| `assets` | 空 | 小数据集借 base 统计量 |
| `base_config` | `None` | 多任务动态 prompt |

### 7.3 平台工厂子类特有旋钮

`LeRobotAlohaDataConfig`([config.py:235](../src/openpi/training/config.py)):`default_prompt`、`use_delta_joint_actions`(默认 True)、`adapt_to_pi`(默认 True)、`repack_transforms`(默认单相机 cam_high←top)、`action_sequence_keys`(默认 `("action",)`)。

`LeRobotLiberoDataConfig`([config.py:288](../src/openpi/training/config.py)):`extra_delta_transform`(默认 False)。

### 7.4 `AssetsConfig`([config.py:37](../src/openpi/training/config.py))

| 字段 | 默认 | 什么时候改 |
|---|---|---|
| `assets_dir` | `None`(回退到 config 默认目录) | 想从 base 目录读 norm_stats |
| `asset_id` | `None`(回退到 `repo_id`) | 借别的机器人统计量 |

### 7.5 `DataConfig` 底层(通过 `base_config=` 改)

| 字段 | 默认 | 什么时候改 |
|---|---|---|
| `prompt_from_task` | `False` | 多任务 |
| `action_sequence_keys` | `("actions",)` | action key 名不同 |
| `use_quantile_norm` | 自动(按 model_type) | **别手设** |

### 7.6 模型配置 `Pi0Config`([pi0_config.py:18](../src/openpi/models/pi0_config.py))

| 字段 | 默认 | 什么时候改 |
|---|---|---|
| `pi05` | `False` | 用 π0.5 设 `True` |
| `action_dim` | `32` | 一般不动,须和 base ckpt 一致(决定 state/actions 填充维度) |
| `action_horizon` | `50` | 改动作序列长度 |
| `max_token_len` | 自动(π0=48, π0.5=200) | 一般不动 |
| `paligemma_variant` | `"gemma_2b"` | LoRA 微调配 `"gemma_2b_lora"` |
| `action_expert_variant` | `"gemma_300m"` | LoRA 微调配带 lora 变体 |

> LoRA 微调关键:`paligemma_variant`/`action_expert_variant` 选带 `lora` 的变体,`freeze_filter=model.get_freeze_filter()` 自动冻结主体只训 LoRA。

---

## 八、推理服务:`serve_policy.py`

[`serve_policy.py`](../scripts/serve_policy.py) 是推理侧**唯一入口**。核心流程([main()](../scripts/serve_policy.py)):

```
命令行参数 →tyro→ Args → create_policy() 构建 Policy →(可选)套 PolicyRecorder → WebsocketPolicyServer → serve_forever()
```

### 8.1 tyro:从 dataclass 自动生成 CLI

[tyro](https://github.com/brentyi/tyro) 是现成的包(openpi 已依赖)。写一个 `@dataclass` 描述参数,`tyro.cli(Args)` 自动变成命令行接口:枚举自动限定取值、`bool` 变 `--flag/--no-flag`、`str|None` 变可选。`--help` 自动生成。

### 8.2 冒号子命令 `policy:checkpoint` / `policy:default`

`Args.policy` 的类型是 `Checkpoint | Default`(Union),tyro 把它做成**互斥子命令**:

- `policy:checkpoint` → 实例化 `Checkpoint`,需要 `--policy.config`、`--policy.dir`
- `policy:default` → 实例化 `Default`,用预置表

冒号 `:` 是 tyro 的**子命令路径分隔符**:左边 `policy` 是字段名,右边 `checkpoint` 是被实例化的 dataclass 名(小写)。用冒号而非空格,是为了支持嵌套子命令、且把子命令明确绑定到具体字段,避免歧义。

子命令自己的参数带 `--policy.` 前缀(因为属于 `policy` 字段):
```
--policy.config  --policy.dir
```

### 8.3 ⚠️ 参数顺序规则(易错)

**全局参数必须在子命令前面,子命令参数在后面。** tyro 把每个 `--xxx` 归属给"正前方的子命令",顺序严格:

```bash
✅ uv run scripts/serve_policy.py --port 9000 --record policy:checkpoint --policy.config=X --policy.dir=Y
   └── 全局参数(前)──┘  └─ 子命令 ─┘  └── 子命令参数(后)──┘

❌ uv run scripts/serve_policy.py policy:checkpoint --policy.config=X --policy.dir=Y --port 9000
   # --port 放后面被当成 checkpoint 子命令参数 → Unrecognized options
```

全局参数:`--port`(默认 8000)、`--record`/`--no-record`、`--env`、`--default-prompt`(注意是**连字符**,不是 `--default.prompt`)。

> flag 命名规则:嵌套层级用**点**(`--policy.config`),字段名下划线转**连字符**(`default_prompt` → `--default-prompt`)。

### 8.4 `policy:default --env` 的含义

`--env` 是 `policy:default` 专用的快捷选择器,从 [`DEFAULT_CHECKPOINT`](../scripts/serve_policy.py) 表挑"配置名+checkpoint":

| `--env=` | 配置 | checkpoint |
|---|---|---|
| `aloha` | `pi05_aloha` | `pi05_base` |
| `aloha_sim` | `pi0_aloha_sim` | `pi0_aloha_sim` |
| `droid` | `pi05_droid` | `pi05_droid` |
| `libero` | `pi05_libero` | `pi05_libero` |

**只有这 4 个平台**(PI 官方发布 ckpt 的)。没有 `ur5` —— UR 必须走 `policy:checkpoint`。

关键:**config 决定用哪套平台变换 + 哪个 ckpt,两者配套**。`--env=aloha` 装配 `AlohaInputs` + `pi05_aloha` 权重,客户端必须发 ALOHA 格式观测;UR 用 `--policy.config=pi05_ur_ping_pong` 装配 UR 变换 + UR 权重,客户端发 UR 格式。模型本身平台无关,平台特异性全在变换,而变换由 config 决定。

### 8.5 `PolicyRecorder`([policy.py:113](../src/openpi/policies/policy.py))

调试用的**包装器**。套在 Policy 外,每次 `infer` 把 `{inputs: obs, outputs: actions}` 存成 `policy_records/step_N.npy`,对服务行为零影响。`--record` 开启,默认关。用 `examples/policy_records.ipynb` 回放。

### 8.6 `WebsocketPolicyServer`([websocket_policy_server.py](../src/openpi/serving/websocket_policy_server.py))

**不需要单独配置**,在 `main()` 里直接实例化:`host="0.0.0.0"`(硬编码)、`port=args.port`、`metadata=policy.metadata`。你只通过 `--port` 间接控制端口,其余改源码。服务端通用,只认 `BasePolicy.infer(obs)`。

协议:客户端连上 → 服务端发 metadata → 循环:收 msgpack 的 obs dict → `policy.infer` → 发回 msgpack 的 action dict(numpy 用 msgpack 序列化)。客户端用 `openpi-client` 的 `WebsocketClientPolicy(host, port)`,不装 JAX,这就是 client/server 分离。

---

## 九、启动命令示例

```bash
cd ~/projects/openpi   # 远程项目目录

# 看全部参数(不带子命令)
uv run scripts/serve_policy.py --help

# 用预置默认策略
uv run scripts/serve_policy.py policy:default --env=aloha_sim
uv run scripts/serve_policy.py --port 9000 --env=droid policy:default

# 用自己的 checkpoint(全局参数在前,子命令参数在后)
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_ur_ping_pong \
  --policy.dir=/data/001_models/pi05_ur_ping_pong/ur_0706_130p_lora/59999/

# 换端口 + 录制 + 默认 prompt
uv run scripts/serve_policy.py \
  --port 9000 --record --default-prompt "hit the ping pong ball back" \
  policy:checkpoint \
  --policy.config=pi05_ur_ping_pong \
  --policy.dir=/data/001_models/.../59999/

# 后台常驻 + 日志
nohup uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_ur_ping_pong --policy.dir=.../59999/ --port 8000 \
  > serve.log 2>&1 &  echo $! > serve.pid
```

客户端连接(Python):
```python
from openpi_client import WebsocketClientPolicy
policy = WebsocketClientPolicy(host="GPU机器IP", port=8000)
action = policy.infer(obs)   # 发观测 → 收 action chunk
```

> ⚠️ 远程那张卡若已被同事的 `serve_policy.py` 占用(~18GB),不要再起;要试先用 `--port 8001` + 确认显存够,且先打招呼。

---

## 十、黄金法则

> **绝大多数字段都有"够用的默认值",你只需要改"和你的数据集/场景对不上的那几个"。** 判断顺序:
> 1. 数据集 key 布局 → `repack`
> 2. 动作绝对/增量 → delta 旋钮
> 3. 指令来源 → `default_prompt` 或 `prompt_from_task`
> 4. norm_stats 来源 → `assets`
> 5. 微调方式 → `weight_loader` + LoRA 变体 + `freeze_filter`
>
> 最稳的实操:**在 `_CONFIGS` 里抄一个最像的 config 当模板,只改这几处。** 不确定就 `--help` 或看工厂源码。

---

## 十一、下一步

- **深入模型**:[pi0.py](../src/openpi/models/pi0.py)(flow matching)、[gemma.py](../src/openpi/models/gemma.py)+[siglip.py](../src/openpi/models/siglip.py)(PaliGemma)、[lora.py](../src/openpi/models/lora.py)。
- **训练循环**:[train.py](../scripts/train.py)+[data_loader.py](../src/openpi/training/data_loader.py)+[checkpoints.py](../src/openpi/training/checkpoints.py)。
- **norm_stats 细节**:[normalize.py](../src/openpi/shared/normalize.py)+[norm_stats.md](../docs/norm_stats.md)。
- **客户端闭环**:[openpi-client/](../packages/openpi-client/src/openpi_client/)(`WebsocketClientPolicy`+`ActionChunkBroker`+`runtime`)。
- **UR5 接入教程**:[examples/ur5/README.md](../examples/ur5/README.md)。
