# Piper Diffusion Policy

这是一个面向松灵 / AgileX Piper 机械臂的真机 Diffusion Policy 仓库。当前版本只保留真机训练、Piper 数据读取、HDF5 到 Zarr 转换、checkpoint 保存、以及真机推理控制相关代码。

策略输入：

```text
camera_wrist   [2, 3, 240, 320]
camera_global  [2, 3, 240, 320]
robot_qpos     [2, 6]
robot_eef_pose [2, 6]
```

策略输出：

```text
未来 16 步 6 关节 qpos，单位 rad
```

真机推理使用 Piper `JointCtrl` 位置控制，执行层使用 50Hz 线性插值和关节限幅来降低动作卡顿和单步跳变。

## 项目结构

```text
.
├── train.py
├── eval_piper_real_robot.py
├── environment.yaml
├── pyproject.toml
├── REAL_DATASET_FORMAT.md
└── diffusion_policy/
    ├── config/
    │   ├── task/
    │   │   ├── piper_real_image.yaml
    │   │   └── piper_zarr_real_image.yaml
    │   ├── train_diffusion_unet_piper_real_image_workspace.yaml
    │   └── train_diffusion_unet_piper_zarr_real_image_workspace.yaml
    ├── dataset/
    │   ├── piper_hdf5_image_dataset.py
    │   └── piper_zarr_image_dataset.py
    ├── scripts/
    │   ├── convert_piper_hdf5_to_zarr.py
    │   ├── check_piper_hdf5_dataset.py
    │   └── check_piper_zarr_dataset.py
    ├── model/
    ├── policy/
    └── workspace/
```

## 安装

如果你已经有可用的 `diffusion` conda 环境，直接安装当前仓库：

```bash
conda activate diffusion
python -m pip install -e .
```

如果需要重新创建环境：

```bash
conda env create -f environment.yaml
conda activate diffusion
python -m pip install -e .
```

安装 Piper SDK：

```bash
python -m pip install -e /home/gx4070/Desktop/arm-datasets-collect/piper_sdk
```

检查环境：

```bash
python -c "import torch, cv2, can, piper_sdk; print('ok')"
```

## 数据

默认 Zarr 数据集路径：

```text
/home/gx4070/Desktop/arm-datasets-collect/data/piper_xy_zarr
```

默认初始位姿文件：

```text
/home/gx4070/Desktop/arm-datasets-collect/initial_pose/initial_pose.json
```

每个 Zarr episode 的格式：

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

从 HDF5 转为 Zarr：

```bash
python diffusion_policy/scripts/convert_piper_hdf5_to_zarr.py \
  --input /home/gx4070/Desktop/arm-datasets-collect/data/piper_xy \
  --output /home/gx4070/Desktop/arm-datasets-collect/data/piper_xy_zarr
```

检查 Zarr 数据：

```bash
python diffusion_policy/scripts/check_piper_zarr_dataset.py
```

正常样本 shape：

```text
obs/camera_wrist:   [2, 3, 240, 320]
obs/camera_global:  [2, 3, 240, 320]
obs/robot_qpos:     [2, 6]
obs/robot_eef_pose: [2, 6]
action:             [16, 6]
```

## 训练

正式训练：

```bash
python train.py \
  --config-name=train_diffusion_unet_piper_zarr_real_image_workspace \
  training.device=cuda:0 \
  logging.mode=offline
```

Debug 训练：

```bash
python train.py \
  --config-name=train_diffusion_unet_piper_zarr_real_image_workspace \
  training.debug=True \
  training.device=cuda:0 \
  logging.mode=offline \
  checkpoint.save_last_ckpt=False \
  checkpoint.topk.k=0
```

关键默认配置：

```text
horizon: 16
n_obs_steps: 2
n_action_steps: 8
batch_size: 32
num_workers: 8
num_epochs: 500
checkpoint_every: 100
```

训练输出默认写到：

```text
data/outputs/YYYY.MM.DD/HH.MM.SS_train_diffusion_unet_piper_zarr_real_image_piper_zarr_real_image/
```

checkpoint 位于：

```text
.../checkpoints/latest.ckpt
.../checkpoints/epoch=XXXX-train_loss=YYYY.ckpt
```

注意：当前 UNet image 模型单个 checkpoint 可能达到数 GB。正式训练前建议预留至少 30GB 可用磁盘。

## 真机推理

先 dry-run，只加载模型、相机、CAN、机械臂反馈，不下发动作：

```bash
python eval_piper_real_robot.py \
  --ckpt /path/to/your/checkpoint.ckpt \
  --initial-pose-json /home/gx4070/Desktop/arm-datasets-collect/initial_pose/initial_pose.json \
  --can can0 \
  --camera-wrist 10 \
  --camera-global 4 \
  --frequency 10 \
  --send-hz 50 \
  --steps-per-inference 2 \
  --speed-percent 10 \
  --max-duration 10 \
  --no-can-judge
```

确认 dry-run 正常后，再低速真机运行：

```bash
python eval_piper_real_robot.py \
  --ckpt /path/to/your/checkpoint.ckpt \
  --initial-pose-json /home/gx4070/Desktop/arm-datasets-collect/initial_pose/initial_pose.json \
  --can can0 \
  --camera-wrist 10 \
  --camera-global 4 \
  --frequency 10 \
  --send-hz 50 \
  --steps-per-inference 8 \
  --speed-percent 40 \
  --reset-duration 5 \
  --max-duration 60 \
  --num-inference-steps 16 \
  --max-joint-delta-rad-per-send 0.03 \
  --max-joint-delta-rad-per-policy-step 0.07 \
  --no-can-judge \
  --enable-motion
```

常用推理参数：

```text
--num-inference-steps
Diffusion 去噪步数。8 更快，16 默认折中，32 更慢但可能更稳。

--max-joint-delta-rad-per-send
50Hz 每次 JointCtrl 下发时，单关节允许变化的最大弧度。

--max-joint-delta-rad-per-policy-step
10Hz policy waypoint 之间，单关节允许变化的最大弧度。
```

停止方式：

```text
q
Esc
Ctrl+C
```

正常停止会让 Piper 进入 standby，不默认 `DisablePiper()`。

## 安全

- 第一次真机运行使用 `--speed-percent 40`。
- 急停按钮必须在手边。
- 先 dry-run，再加 `--enable-motion`。
- 确认 `camera_wrist` 和 `camera_global` 没有接反。
- 确认 `initial_pose.json` 的 reset 位姿安全可达。
- 如果真机轨迹方向或速度异常，立即停止。
