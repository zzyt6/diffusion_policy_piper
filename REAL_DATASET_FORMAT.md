# Piper 真实机械臂数据集格式

这份文档说明当前仓库真正支持的数据格式。目标是：你整理好数据后，可以直接训练 Piper 真机 Diffusion Policy。

当前仓库支持两条数据路径：

```text
1. HDF5 原始 episode 直读：适合检查原始采集结果，训练时会随机读 HDF5 并 resize 图像。
2. Zarr 预处理 episode：推荐正式训练，图像已经 resize 到 240x320，训练读取更快。
```

默认路径：

```text
HDF5: /home/gx4070/Desktop/arm-datasets-collect/data/piper_xy
Zarr: /home/gx4070/Desktop/arm-datasets-collect/data/piper_xy_zarr
```

## 训练任务定义

模型输入 observation：

```text
camera_wrist    wrist 腕部相机图像
camera_global   全局相机图像
robot_qpos      当前 6 关节位置，单位 rad
robot_eef_pose  当前末端位姿，当前数据为 mm + deg
```

模型监督 action：

```text
action          下一步 6 关节 qpos，单位 rad
```

当前配置里的训练样本 shape：

```text
obs/camera_wrist:    [2, 3, 240, 320]  float32
obs/camera_global:   [2, 3, 240, 320]  float32
obs/robot_qpos:      [2, 6]            float32
obs/robot_eef_pose:  [2, 6]            float32
action:              [16, 6]           float32
```

含义：

```text
n_obs_steps = 2     使用当前和上一帧观测
horizon = 16        训练目标是连续 16 步未来关节位置
action_dim = 6      Piper 6 个关节 qpos
```

注意：训练时使用 Zarr/HDF5 文件；真机推理时不读取 Zarr。真机推理直接使用实时相机图像和机械臂反馈，并做同样的 resize、CHW 转换和归一化。

## HDF5 原始格式

每个 `.hdf5` 文件是一条 episode：

```text
piper_xy/
├── episode_000000.hdf5
├── episode_000001.hdf5
└── ...
```

推荐 schema：

```text
piper_diffusion_joint_action_hdf5_v2
```

所有 dataset 第一维都是时间步 `T`。

### 必需字段

```text
observations/images/wrist     uint8   [T, H, W, 3]
observations/images/global    uint8   [T, H, W, 3]
observations/qpos             float32 [T, 6]
observations/eef_pose         float32 [T, 6]
action                        float32 [T, 6]
time/timestamp_ns             int64   [T]
valid/wrist_camera            bool    [T]
valid/global_camera           bool    [T]
valid/robot_feedback          bool    [T]
valid/action                  bool    [T]
```

字段含义：

```text
observations/images/wrist   腕部相机 RGB 图像，HWC，uint8
observations/images/global  全局相机 RGB 图像，HWC，uint8
observations/qpos           当前反馈 6 关节角，rad
observations/eef_pose       当前末端位姿，[x,y,z,rx,ry,rz]，当前数据为 mm + deg
action                      下一步 6 关节角，rad
time/timestamp_ns           采样时刻，纳秒
```

动作对齐关系：

```text
obs[t]    = images[t] + qpos[t] + eef_pose[t]
action[t] = qpos[t + 1]
```

最后一帧通常没有下一帧作为监督目标，所以：

```text
valid/action[-1] = false
```

### 可选字段

这些字段可以保留，但当前训练不会读取：

```text
observations/state                 float32 [T, 12]  eef_pose + qpos
actual_action                      float32 [T, 6]
actions/actual_qpos_next_rad       float32 [T, 6]
actions/actual_joint_delta_rad     float32 [T, 6]
actions/actual_delta_xy_mm         float32 [T, 2]
actions/actual_eef_pose_next_mm_deg float32 [T, 6]
actions/command_delta_xy_mm        float32 [T, 2]
actions/command_eef_pose_mm_deg    float32 [T, 6]
actions/command_sdk_units          int32   [T, 6]
actions/command_count              int32   [T]
control/key_state                  bool    [T, 4]
control/xy_direction               float32 [T, 2]
valid/actual_action                bool    [T]
valid/command_sent                 bool    [T]
```

这些字段主要用于审计、调试和分析采集质量。

### HDF5 attributes

建议保留这些 metadata：

```text
schema_version
created_at
control_hz
command_hz
dt_seconds
command_dt_seconds
alignment
action_units
eef_pose_units
qpos_units
image_encoding
joint_names_json
pose_names_json
key_names_json
config_json
start_pose_mm_deg_json
initial_pose_source
initial_pose_json
work_plane_json
reset_json
```

当前训练代码不强依赖这些 attributes，但它们对追踪数据来源很重要。

## Zarr 预处理格式

正式训练推荐使用 Zarr。每个 `.zarr` 目录是一条 episode：

```text
piper_xy_zarr/
├── episode_000000.zarr/
├── episode_000001.zarr/
└── ...
```

每个 episode 内部必须包含：

```text
episode_xxx.zarr/
├── camera_wrist       uint8   [T, 240, 320, 3]
├── camera_global      uint8   [T, 240, 320, 3]
├── robot_qpos         float32 [T, 6]
├── robot_eef_pose     float32 [T, 6]
├── action             float32 [T, 6]
├── valid              bool    [T]
└── timestamp          float64 [T]
```

转换规则：

```text
camera_wrist    <- observations/images/wrist，resize 到 240x320
camera_global   <- observations/images/global，resize 到 240x320
robot_qpos      <- observations/qpos
robot_eef_pose  <- observations/eef_pose
action          <- action
valid           <- valid/wrist_camera & valid/global_camera & valid/robot_feedback & valid/action
timestamp       <- time/timestamp_ns / 1e9
```

额外校验：

```text
action、robot_qpos、robot_eef_pose 必须全部是 finite 数值
valid.sum() 必须大于 0
所有字段 shape 和 dtype 必须匹配
```

图像 chunk：

```text
(1, 240, 320, 3)
```

低维数据 chunk：

```text
(min(T, 4096), dim)
```

压缩：

```text
numcodecs.Blosc(cname="zstd")
```

## HDF5 转 Zarr

全量转换：

```bash
python diffusion_policy/scripts/convert_piper_hdf5_to_zarr.py \
  --input /home/gx4070/Desktop/arm-datasets-collect/data/piper_xy \
  --output /home/gx4070/Desktop/arm-datasets-collect/data/piper_xy_zarr
```

先转换前 5 条检查：

```bash
python diffusion_policy/scripts/convert_piper_hdf5_to_zarr.py \
  --input /home/gx4070/Desktop/arm-datasets-collect/data/piper_xy \
  --output /home/gx4070/Desktop/arm-datasets-collect/data/piper_xy_zarr \
  --limit 5
```

覆盖已有 Zarr：

```bash
python diffusion_policy/scripts/convert_piper_hdf5_to_zarr.py \
  --input /home/gx4070/Desktop/arm-datasets-collect/data/piper_xy \
  --output /home/gx4070/Desktop/arm-datasets-collect/data/piper_xy_zarr \
  --overwrite
```

默认不会删除源 HDF5。只有明确加上 `--delete-source`，且单条 Zarr 校验通过后，脚本才会删除对应 HDF5：

```bash
python diffusion_policy/scripts/convert_piper_hdf5_to_zarr.py \
  --input /home/gx4070/Desktop/arm-datasets-collect/data/piper_xy \
  --output /home/gx4070/Desktop/arm-datasets-collect/data/piper_xy_zarr \
  --delete-source
```

不建议在第一次转换时使用 `--delete-source`。

## Dataset 采样逻辑

训练时会扫描所有 episode，按文件名排序。

对每个 episode：

```text
sequence_length = horizon + n_latency_steps
当前默认 sequence_length = 16
```

只有当一个连续窗口内所有帧 `valid == true` 时，这个窗口才会成为训练样本。

单个样本：

```text
obs_start = start_idx
obs_end   = start_idx + n_obs_steps
act_start = start_idx
act_end   = start_idx + horizon
```

返回：

```python
{
    "obs": {
        "camera_wrist": Tensor[n_obs_steps, 3, 240, 320],
        "camera_global": Tensor[n_obs_steps, 3, 240, 320],
        "robot_qpos": Tensor[n_obs_steps, 6],
        "robot_eef_pose": Tensor[n_obs_steps, 6],
    },
    "action": Tensor[horizon, 6],
}
```

如果设置了 `n_latency_steps > 0`，动作会跳过前面的 latency 步。

## 图像和归一化

HDF5 图像原始格式：

```text
uint8 [T, H, W, 3]
```

Zarr 图像格式：

```text
uint8 [T, 240, 320, 3]
```

Dataset 返回给网络前会转成：

```text
float32 [T, 3, 240, 320] / 255.0
```

Normalizer：

```text
camera_wrist/camera_global: image range normalizer
robot_qpos:                 range normalizer
robot_eef_pose:             range normalizer
action:                     range normalizer
```

## 检查数据

检查 HDF5：

```bash
python diffusion_policy/scripts/check_piper_hdf5_dataset.py
```

检查 Zarr：

```bash
python diffusion_policy/scripts/check_piper_zarr_dataset.py
```

正常 Zarr 检查输出应包含类似：

```text
episodes: 66
train windows: 33328
rgb keys: ['camera_wrist', 'camera_global']
lowdim keys: ['robot_qpos', 'robot_eef_pose']
obs/camera_wrist: shape=(2, 3, 240, 320)
obs/camera_global: shape=(2, 3, 240, 320)
obs/robot_qpos: shape=(2, 6)
obs/robot_eef_pose: shape=(2, 6)
action: shape=(16, 6)
```

## 训练配置

Zarr 正式训练：

```bash
python train.py \
  --config-name=train_diffusion_unet_piper_zarr_real_image_workspace \
  training.device=cuda:0 \
  logging.mode=offline
```

Zarr debug：

```bash
python train.py \
  --config-name=train_diffusion_unet_piper_zarr_real_image_workspace \
  training.debug=True \
  training.device=cuda:0 \
  logging.mode=offline \
  checkpoint.save_last_ckpt=False \
  checkpoint.topk.k=0
```

HDF5 直读训练：

```bash
python train.py \
  --config-name=train_diffusion_unet_piper_real_image_workspace \
  training.device=cuda:0 \
  logging.mode=offline
```

配置文件：

```text
diffusion_policy/config/task/piper_zarr_real_image.yaml
diffusion_policy/config/task/piper_real_image.yaml
diffusion_policy/config/train_diffusion_unet_piper_zarr_real_image_workspace.yaml
diffusion_policy/config/train_diffusion_unet_piper_real_image_workspace.yaml
```

修改数据路径：

```bash
python train.py \
  --config-name=train_diffusion_unet_piper_zarr_real_image_workspace \
  task.dataset_path=/your/zarr/path \
  training.device=cuda:0 \
  logging.mode=offline
```

## 相机对应关系

当前命名固定为：

```text
camera_wrist   腕部相机，来自 observations/images/wrist 或 zarr camera_wrist
camera_global  全局相机，来自 observations/images/global 或 zarr camera_global
```

训练和真机推理必须保持这个语义一致。不要把 wrist 和 global 反过来训练，推理时也不要把两个相机编号传反。

## 推理时的数据格式

真机推理不需要 Zarr 文件。推理脚本实时构造与训练一致的 observation：

```text
camera_wrist    实时 wrist 相机图像 -> RGB -> resize 240x320 -> CHW -> /255
camera_global   实时 global 相机图像 -> RGB -> resize 240x320 -> CHW -> /255
robot_qpos      Piper 当前反馈 qpos，rad
robot_eef_pose  Piper 当前反馈末端位姿
```

模型输出：

```text
action [n_action_steps, 6]
```

它表示未来若干步绝对关节 qpos，单位 rad。执行脚本会做关节限幅和 50Hz 插值后下发 `JointCtrl`。

## 自己重新采集时的最低要求

如果你重新写采集脚本，至少保证每条 episode 能提供：

```text
两路 RGB 图像
当前 qpos
当前 eef_pose
下一步 qpos action
每帧 timestamp
每帧 valid mask
episode 边界，最好一个文件一个 episode
```

如果直接生成 Zarr，可以跳过 HDF5，但必须完全符合上面的 Zarr schema。

## 常见错误

```text
错误 1：action 写成当前 qpos，而不是下一步 qpos。
结果：模型学到保持不动。

错误 2：wrist/global 相机语义训练和推理反了。
结果：模型看到的图像条件和训练不一致。

错误 3：最后一帧 action 没有标 invalid。
结果：训练会吃到错误监督。

错误 4：图像是 BGR，但按 RGB 训练。
结果：颜色分布不一致。

错误 5：qpos 单位不是 rad。
结果：动作尺度完全错误。

错误 6：Zarr 数据有坏帧，但 valid 没有过滤。
结果：训练窗口会包含无效图像或无效动作。
```
