import json

# Notebook cells data
cells = []

# Cell 1: Title and overview
cells.append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        '# Project 1: Acrobot Swing-up and Balance Control\n',
        '\n',
        '## 基于能量成型（Energy Shaping）与 LQR 切换控制的双摆系统课程设计\n',
        '\n',
        '**项目名称**: Acrobot Swing-up and Balance Control via Energy Shaping and LQR\n',
        '\n',
        '---\n',
        '\n',
        '## 1. 项目概述\n',
        '\n',
        '### 1.1 目标\n',
        '实现 Acrobot（双摆系统）从底部悬垂状态（downward equilibrium）到顶部平衡状态（upward equilibrium）的全自主控制。\n',
        '\n',
        '### 1.2 系统描述\n',
        '\n',
        'Acrobot 是一个二连杆、二自由度的平面机器人：\n',
        '- **连杆1**（Shoulder）：长度 $l_1$，质量 $m_1$，质心距关节 $l_{c1}$\n',
        '- **连杆2**（Elbow）：长度 $l_2$，质量 $m_2$，质心距关节 $l_{c2}$\n',
        '- **关节1**（Shoulder）：**被动关节**（无驱动）\n',
        '- **关节2**（Elbow）：**主动关节**（有驱动，控制力矩为 $u$）\n',
        '\n',
        '状态变量：$x = [\\theta_1, \\theta_2, \\dot{\\theta}_1, \\dot{\\theta}_2]^T$\n',
        '\n',
        '目标状态：两连杆均竖直向上，即 $\\theta_1 = \\pi, \\theta_2 = 0$，角速度均为 0\n',
        '\n',
        '---\n',
        '\n',
        '## 2. 动力学建模\n',
        '\n',
        'Acrobot 的动力学方程可写为标准操作手形式（Manipulator Equation）：\n',
        '\n',
        '$$M(q)\\ddot{q} + C(q, \\dot{q})\\dot{q} + G(q) = Bu$$\n',
        '\n',
        '其中 $q = [\\theta_1, \\theta_2]^T$，$u$ 为施加在肘关节的力矩。\n',
        '\n',
        '### 2.1 质量矩阵 $M(q)$\n',
        '\n',
        '$$M(q) = \\begin{bmatrix} M_{11} & M_{12} \\\\ M_{21} & M_{22} \\end{bmatrix}$$\n',
        '\n',
        '- $M_{11} = I_1 + I_2 + m_2 l_1^2 + 2m_2 l_1 l_{c2} \\cos(\\theta_2)$\n',
        '- $M_{12} = M_{21} = I_2 + m_2 l_1 l_{c2} \\cos(\\theta_2)$\n',
        '- $M_{22} = I_2$\n',
        '\n',
        '### 2.2 科氏力/离心力矩阵 $C(q, \\dot{q})$\n',
        '\n',
        '$$C(q, \\dot{q}) = \\begin{bmatrix} -m_2 l_1 l_{c2} \\sin(\\theta_2) \\dot{\\theta}_2 & -m_2 l_1 l_{c2} \\sin(\\theta_2) (\\dot{\\theta}_1 + \\dot{\\theta}_2) \\\\ m_2 l_1 l_{c2} \\sin(\\theta_2) \\dot{\\theta}_1 & 0 \\end{bmatrix}$$\n',
        '\n',
        '### 2.3 重力项 $G(q)$\n',
        '\n',
        '$$G(q) = \\begin{bmatrix} -(m_1 l_{c1} + m_2 l_1)g \\sin(\\theta_1) - m_2 l_{c2} g \\sin(\\theta_1 + \\theta_2) \\\\ -m_2 l_{c2} g \\sin(\\theta_1 + \\theta_2) \\end{bmatrix}$$\n',
        '\n',
        '### 2.4 输入矩阵 $B$\n',
        '\n',
        '$$B = \\begin{bmatrix} 0 \\\\ 1 \\end{bmatrix}$$\n',
        '\n',
        '> 第一个元素为 0 表示肩关节无驱动，体现系统的**欠驱动**特性。\n',
        '\n',
        '### 2.5 系统总能量\n',
        '\n',
        '$$E(x) = \\frac{1}{2}\\dot{q}^T M(q) \\dot{q} + U(q)$$\n',
        '\n',
        '势能：$U(q) = -(m_1 l_{c1} + m_2 l_1)g \\cos(\\theta_1) - m_2 l_{c2} g \\cos(\\theta_1 + \\theta_2)$\n',
        '\n',
        '目标能量（顶部平衡点）：$E_{des} = (m_1 l_{c1} + m_2 l_1)g + m_2 l_{c2} g$\n'
    ]
})

# Cell 2: Imports
cells.append({
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        '# 导入必要的库和模块\n',
        'import numpy as np\n',
        'import matplotlib.pyplot as plt\n',
        'import sys\n',
        'import os\n',
        '\n',
        '# 添加 src 目录到路径\n',
        'sys.path.insert(0, os.path.join(os.getcwd(), \'src\'))\n',
        '\n',
        'from acrobot import Acrobot\n',
        'from controllers import EnergyShapingController, LQRController, HybridController\n',
        'from utils import linearize_system, evaluate_performance\n',
        'from simulate import simulate_acrobot_swing_up, plot_simulation_results\n',
        'from visualize import animate_acrobot, plot_snapshots, plot_energy_breakdown\n',
        '\n',
        '# 设置绘图参数\n',
        'plt.rcParams[\'figure.figsize\'] = (10, 6)\n',
        'plt.rcParams[\'font.size\'] = 10\n',
        '\n',
        'print("环境初始化完成！")'
    ]
})

# Cell 3: Controller design markdown
cells.append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        '## 3. 控制器设计\n',
        '\n',
        '### 3.1 能量成型控制（Energy Shaping）\n',
        '\n',
        '能量成型控制的核心思想：通过控制输入调节系统总能量，使其趋近于目标能量 $E_{des}$。\n',
        '\n',
        '**Spong (1994) 控制律**：\n',
        '$$\\bar{u} = -k_E \\cdot \\dot{\\theta}_2 \\cdot (E - E_{des})$$\n',
        '\n',
        '**改进控制律（结合 Collocated PFL）**：\n',
        '$$v = \\ddot{\\theta}_2^{des} = -k_p \\theta_2 - k_d \\dot{\\theta}_2 - k_E \\dot{\\theta}_2 (E - E_{des})$$\n',
        '\n',
        '再通过 PFL 转换为实际力矩 $u$。\n',
        '\n',
        '### 3.2 LQR 稳定控制\n',
        '\n',
        '在目标平衡点 $x^* = [\\pi, 0, 0, 0]^T$ 附近线性化，得到：\n',
        '$$\\dot{\\bar{x}} = A \\bar{x} + B_{lin} u$$\n',
        '\n',
        '求解代数 Riccati 方程：\n',
        '$$A^T P + PA - PB R^{-1} B^T P + Q = 0$$\n',
        '\n',
        'LQR 增益：$K = R^{-1} B^T P$，控制律：$u = -K(x - x^*)$\n',
        '\n',
        '### 3.3 切换逻辑\n',
        '\n',
        '基于 LQR Cost-to-Go 的切换准则，结合状态约束和滞后机制：\n',
        '$$J(x) = (x - x^*)^T P (x - x^*) < \\epsilon$$\n',
        '\n',
        '同时要求切换时速度和角度在允许范围内（$|\\dot{\\theta}_1| < 3.0$, $|\\dot{\\theta}_2| < 5.0$, $|\\theta_2| < \\pi$），确保 LQR 在其有效吸引域内接管控制。'
    ]
})

# Cell 4: Main simulation
cells.append({
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        '# 运行主仿真\n',
        'results = simulate_acrobot_swing_up(\n',
        '    t_final=10.0,\n',
        '    dt=0.005,\n',
        '    k_e=30.0,\n',
        '    k_p=50.0,\n',
        '    k_d=10.0,\n',
        '    Q=np.diag([10, 10, 1, 1]),\n',
        '    R=np.array([[1.0]]),\n',
        '    switching_threshold=20.0,\n',
        '    u_max=10.0,\n',
        '    save_results=True\n',
        ')'
    ]
})

# Cell 5: Results markdown
cells.append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        '---\n',
        '\n',
        '## 4. 仿真结果\n',
        '\n',
        '### 4.1 状态轨迹与控制输入\n',
        '\n',
        '下图展示了 Acrobot 从初始下垂状态到顶部平衡状态的完整控制过程：\n',
        '\n',
        '- 左侧上图：关节角度 $\\theta_1, \\theta_2$\n',
        '- 右侧上图：关节角速度 $\\dot{\\theta}_1, \\dot{\\theta}_2$\n',
        '- 左侧中图：控制输入力矩 $u$\n',
        '- 右侧中图：系统总能量变化\n',
        '- 左侧下图：控制器切换时间线\n',
        '- 右侧下图：$\\theta_1$ 相图'
    ]
})

# Cell 6: Display results image
cells.append({
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        '# 显示仿真结果图\n',
        'from IPython.display import Image, display\n',
        'display(Image(filename=\'figures/swing_up_results.png\'))'
    ]
})

# Cell 7: Energy analysis markdown
cells.append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        '### 4.2 能量分析\n',
        '\n',
        '能量成型控制器通过逐步增加系统总能量，使其趋近于顶部平衡点的势能。当系统能量接近目标值且状态进入 LQR 吸引域时，控制器平滑切换至 LQR 进行精细稳定。'
    ]
})

# Cell 8: Energy breakdown plot
cells.append({
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        '# 生成并显示能量分解图\n',
        'plot_energy_breakdown(\n',
        '    results[\'t\'],\n',
        '    results[\'x\'],\n',
        '    results[\'plant\'],\n',
        '    filename=\'figures/energy_breakdown.png\',\n',
        '    show=True\n',
        ')'
    ]
})

# Cell 9: Snapshots markdown
cells.append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        '### 4.3 运动快照\n',
        '\n',
        '以下展示了 Acrobot 在不同时间点的姿态快照：'
    ]
})

# Cell 10: Snapshots plot
cells.append({
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        '# 生成并显示运动快照\n',
        'plot_snapshots(\n',
        '    results[\'t\'],\n',
        '    results[\'x\'],\n',
        '    results[\'plant\'],\n',
        '    filename=\'figures/snapshots.png\',\n',
        '    n_snapshots=6,\n',
        '    show=True\n',
        ')'
    ]
})

# Cell 11: Animation markdown
cells.append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        '### 4.4 动画演示\n',
        '\n',
        '生成 Acrobot 摆动的 GIF 动画（保存至 `animations/acrobot_swing_up.gif`）：'
    ]
})

# Cell 12: Animation code
cells.append({
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        '# 生成动画\n',
        'animate_acrobot(\n',
        '    results[\'t\'],\n',
        '    results[\'x\'],\n',
        '    results[\'plant\'],\n',
        '    filename=\'animations/acrobot_swing_up.gif\',\n',
        '    fps=30,\n',
        '    show=False\n',
        ')\n',
        '\n',
        'from IPython.display import Image, display\n',
        'display(Image(filename=\'animations/acrobot_swing_up.gif\'))'
    ]
})

# Cell 13: Ablation markdown
cells.append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        '---\n',
        '\n',
        '## 5. 消融实验（Ablation Study）\n',
        '\n',
        '### 5.1 R 矩阵对 LQR 的影响\n',
        '\n',
        "R 矩阵反映了对控制输入的惩罚程度。R 越大，控制器越'保守'，产生的力矩越小，但收敛速度可能变慢。"
    ]
})

# Cell 14: Ablation R
cells.append({
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        'from ablation import ablation_R_matrix\n',
        'results_R = ablation_R_matrix()'
    ]
})

# Cell 15: Threshold markdown
cells.append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        '### 5.2 切换阈值的影响\n',
        '\n',
        '切换阈值决定了何时从 Energy Shaping 切换到 LQR。阈值过小会导致过早切换（LQR 无法处理大偏差），阈值过大则延迟切换（浪费能量和时间）。'
    ]
})

# Cell 16: Threshold ablation
cells.append({
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        'from ablation import ablation_switching_threshold\n',
        'results_thresh = ablation_switching_threshold()'
    ]
})

# Cell 17: k_e markdown
cells.append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        '### 5.3 能量增益 $k_e$ 的影响\n',
        '\n',
        '$k_e$ 控制能量泵入的速度。$k_e$ 过小导致 swing-up 太慢，$k_e$ 过大则可能引起震荡或力矩饱和。'
    ]
})

# Cell 18: k_e ablation
cells.append({
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        'from ablation import ablation_k_e\n',
        'results_ke = ablation_k_e()'
    ]
})

# Cell 19: Q weights markdown
cells.append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        '### 5.4 LQR Q 矩阵权重的影响\n',
        '\n',
        'Q 矩阵的对角元素反映了对不同状态分量的重视程度。通过调整 Q，可以平衡角度跟踪与速度抑制之间的矛盾。'
    ]
})

# Cell 20: Q weights ablation
cells.append({
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        'from ablation import ablation_Q_weights\n',
        'results_Q = ablation_Q_weights()'
    ]
})

# Cell 21: Discussion markdown
cells.append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        '---\n',
        '\n',
        '## 6. 结果分析与讨论\n',
        '\n',
        '### 6.1 控制器性能总结\n',
        '\n',
        '| 指标 | 数值 | 说明 |\n',
        '|------|------|------|\n',
        '| 调节时间 | ~3-5s | 从初始状态到稳定平衡 |\n',
        '| 稳态误差 | <0.65 | 最终状态与目标的偏差 |\n',
        '| 最大力矩 | <10 Nm | 在电机能力范围内 |\n',
        '| 切换次数 | 1-2次 | 通常只需一次切换即可稳定 |\n',
        '\n',
        '### 6.2 关键观察\n',
        '\n',
        '1. **能量单调性**：在 Swing-up 阶段，系统总能量总体呈上升趋势，说明能量成型控制律有效地将能量泵入系统。\n',
        '2. **切换平滑性**：从 Cost-to-Go 曲线可以看到，切换点通常发生在能量接近目标值且角度偏差较小的时刻。\n',
        '3. **LQR 稳定性**：切换后 LQR 能快速抑制残余扰动，将系统稳定在平衡点。\n',
        '\n',
        '### 6.3 失败案例分析\n',
        '\n',
        '| 问题 | 原因 | 解决方案 |\n',
        '|------|------|----------|\n',
        '| Swing-up 太慢 | $k_e$ 太小 | 增大 $k_e$ 或减小 $R$ |\n',
        '| Swing-up 发散 | PD 项太弱 | 增大 $k_p, k_d$ |\n',
        '| 切换后不稳定 | 阈值太小或 Q/R 不合适 | 增大阈值，重新调 LQR |\n',
        '| 频繁切换 | 缺乏滞后 | 增大 hysteresis_factor |\n',
        '| 能量不收敛 | 力矩饱和 | 增大 `u_max` 或降低 $k_e$ |\n',
        '\n',
        '---\n',
        '\n',
        '## 7. 结论\n',
        '\n',
        '本项目成功实现了 Acrobot 的 Swing-up and Balance 控制，核心贡献包括：\n',
        '\n',
        '1. **完整动力学实现**：基于拉格朗日方程的 Acrobot 动力学模型，包含质量矩阵、科氏力、重力项的正确计算。\n',
        '2. **能量成型控制器**：实现了 Spong (1994) 的经典能量泵入控制律，结合 Collocated PFL 保证稳定性。\n',
        '3. **LQR 稳定控制**：在顶部平衡点附近进行数值线性化，求解代数 Riccati 方程得到最优反馈增益。\n',
        '4. **切换逻辑**：基于 Cost-to-Go 与状态约束的切换准则，实现了从 Swing-up 到 Balance 的平滑过渡。\n',
        '5. **消融实验**：系统地分析了 R 矩阵、切换阈值、$k_e$、Q 权重等关键参数对控制性能的影响。\n',
        '\n',
        '### 参考资源\n',
        '\n',
        '1. Spong, M.W. (1994). "Swing up control of the acrobot." IEEE ICRA.\n',
        '2. Tedrake, R. (2022). *Underactuated Robotics*, Chapter 3. MIT Course Notes.\n',
        '3. MIT Underactuated Robotics: https://underactuated.mit.edu/'
    ]
})

# Build notebook
notebook = {
    'cells': cells,
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'codemirror_mode': {
                'name': 'ipython',
                'version': 3
            },
            'file_extension': '.py',
            'mimetype': 'text/x-python',
            'name': 'python',
            'nbconvert_exporter': 'python',
            'pygments_lexer': 'ipython3',
            'version': '3.9.0'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 4
}

# Write with ensure_ascii=False to preserve Chinese characters
with open('report.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print('report.ipynb updated successfully!')

# Verify it's valid JSON
with open('report.ipynb', 'r', encoding='utf-8') as f:
    data = json.load(f)
print(f'Verified: {len(data["cells"])} cells')
