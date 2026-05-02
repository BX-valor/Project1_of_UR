# Project 1: Acrobot Swing-up and Balance Control

基于能量成型（Energy Shaping）与 LQR 切换控制的双摆系统课程设计。

## 项目概述

本项目实现 **Acrobot（双摆系统）** 从底部悬垂状态（downward equilibrium）到顶部平衡状态（upward equilibrium）的全自主控制。

Acrobot 是一个二连杆、二自由度的平面机器人：
- **连杆1**（Shoulder）：被动关节（无驱动）
- **连杆2**（Elbow）：主动关节（有驱动，控制力矩为 $u$）

核心方法：
1. **能量成型控制（Energy Shaping）**：基于 Spong (1994) 的经典能量泵入控制律，通过 Collocated PFL 转换为实际力矩
2. **LQR 稳定控制**：在顶部平衡点附近进行数值线性化，求解代数 Riccati 方程得到最优反馈增益
3. **混合切换控制**：基于 LQR Cost-to-Go 与状态约束的切换准则，实现从 Swing-up 到 Balance 的平滑过渡

## 文件结构

```
acrobot_project/
├── src/                          # 源代码
│   ├── acrobot.py               # Acrobot 动力学模型（拉格朗日方程）
│   ├── controllers.py           # 控制器实现（Energy Shaping / LQR / Hybrid）
│   ├── simulate.py              # 主仿真脚本
│   ├── utils.py                 # 工具函数（线性化、性能评估）
│   ├── visualize.py             # 可视化与动画生成
│   └── ablation.py              # 消融实验
├── figures/                      # 生成的图片
│   ├── swing_up_results.png
│   ├── energy_breakdown.png
│   ├── snapshots.png
│   ├── ablation_R_matrix.png
│   ├── ablation_switching_threshold.png
│   ├── ablation_k_e.png
│   └── ablation_Q_weights.png
├── animations/                   # 生成的动画
│   └── acrobot_swing_up.gif
├── report.ipynb                  # Jupyter Notebook 报告
├── generate_report.py            # 生成 report.ipynb 的脚本
├── requirements.txt              # Python 依赖
└── README.md                     # 本文件
```

## 环境要求

- Python >= 3.9
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- Matplotlib >= 3.4.0
- Jupyter Notebook / JupyterLab（可选，用于运行 report.ipynb）

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行主仿真

```bash
cd src
python simulate.py
```

运行后会在 `../figures/` 目录下生成 `swing_up_results.png`。

### 3. 运行消融实验

```bash
cd src
python ablation.py
```

会依次运行以下消融实验并在 `../figures/` 下保存结果：
- R 矩阵对 LQR 的影响
- 切换阈值的影响
- 能量增益 $k_e$ 的影响
- LQR Q 矩阵权重的影响

### 4. 生成动画与快照

```bash
cd src
python visualize.py
```

会生成：
- `../animations/acrobot_swing_up.gif` —— Acrobot 摆动动画
- `../figures/snapshots.png` —— 运动快照
- `../figures/energy_breakdown.png` —— 能量分解图

### 5. 查看 Notebook 报告

```bash
jupyter notebook report.ipynb
```

或

```bash
jupyter lab report.ipynb
```

## 核心算法说明

### 动力学模型

Acrobot 的动力学方程为标准操作手形式：

$$M(q)\ddot{q} + C(q, \dot{q})\dot{q} + G(q) = Bu$$

其中 $q = [\theta_1, \theta_2]^{T}$，$u$ 为施加在肘关节的力矩。$B = [0, 1]^{T}$ 体现系统的**欠驱动**特性。

### 能量成型控制器

Spong (1994) 控制律：

$$\bar{u} = -k_E \cdot \dot{\theta}_2 \cdot (E - E_{des})$$

改进控制律（结合 Collocated PFL）：

$$v = \ddot{\theta}_2^{des} = -k_p \theta_2 - k_d \dot{\theta}_2 - k_E \dot{\theta}_2 (E - E_{des})$$

### LQR 控制器

在目标平衡点 $x^{*} = [\pi, 0, 0, 0]^{T}$ 附近线性化，求解代数 Riccati 方程：

$$A^{T} P + PA - PB R^{-1} B^{T} P + Q = 0$$

LQR 增益：$K = R^{-1} B^{T} P$，控制律：$u = -K(x - x^{*})$

### 切换逻辑

基于 LQR Cost-to-Go 的切换准则：

$$J(x) = (x - x^{*})^{T} P (x - x^{*}) < \epsilon$$

同时要求切换时速度和角度在允许范围内，确保 LQR 在其有效吸引域内接管控制。

## 实验结果

### 典型性能指标

| 指标 | 数值 | 说明 |
|------|------|------|
| 调节时间 | ~3-5 s | 从初始状态到稳定平衡 |
| 稳态误差 | < 0.65 | 最终状态与目标的偏差 |
| 最大力矩 | < 10 Nm | 在电机能力范围内 |
| 切换次数 | 1-2 次 | 通常只需一次切换即可稳定 |

### 消融实验结论

1. **R 矩阵**：R 越大，控制器越保守，力矩越小，收敛速度变慢
2. **切换阈值**：阈值过小导致过早切换（LQR 无法处理大偏差），阈值过大则延迟切换
3. **能量增益 $k_e$**：过小导致 swing-up 太慢，过大则可能引起震荡或力矩饱和
4. **Q 矩阵权重**：不同状态分量的重视程度影响角度跟踪与速度抑制的平衡

## 参考文献

1. Spong, M.W. (1994). "Swing up control of the acrobot." *IEEE International Conference on Robotics and Automation (ICRA)*.
2. Tedrake, R. (2022). *Underactuated Robotics*, Chapter 3. MIT Course Notes.
3. MIT Underactuated Robotics: https://underactuated.mit.edu/

## 作者

- **GitHub**: [BX-valor](https://github.com/BX-valor)
- **项目地址**: https://github.com/BX-valor/Project1_of_UR

## 许可

本项目仅供学术参考与学习交流使用。
