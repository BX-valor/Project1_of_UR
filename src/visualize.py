"""
visualize.py - 可视化与动画生成

包含:
- 生成 Acrobot 摆动的 GIF 动画
- 绘制摆杆运动的快照图
- 能量-时间 3D 可视化（可选）
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def animate_acrobot(t, states, plant, filename='../animations/acrobot_swing_up.gif',
                    fps=30, show=False):
    """
    生成 Acrobot 摆动的 GIF 动画
    
    参数:
        t: 时间数组
        states: 状态轨迹 (4, N)
        plant: Acrobot 实例
        filename: 输出文件名
        fps: 帧率
        show: 是否显示动画（在 Notebook 中可设为 True）
    
    返回:
        anim: FuncAnimation 对象
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(7, 7))
    
    # 计算摆杆轨迹范围以自动调整坐标轴
    all_positions = []
    for i in range(states.shape[1]):
        _, (x1, y1), (x2, y2) = plant.get_cartesian_positions(states[:, i])
        all_positions.extend([(x1, y1), (x2, y2)])
    
    all_x = [p[0] for p in all_positions]
    all_y = [p[1] for p in all_positions]
    margin = 0.5
    xlim = [min(all_x) - margin, max(all_x) + margin]
    ylim = [min(all_y) - margin, max(all_y) + margin]
    
    # 保证坐标轴比例一致且对称
    max_range = max(xlim[1] - xlim[0], ylim[1] - ylim[0]) / 2
    center_x = (xlim[0] + xlim[1]) / 2
    center_y = (ylim[0] + ylim[1]) / 2
    
    ax.set_xlim(center_x - max_range, center_x + max_range)
    ax.set_ylim(center_y - max_range, center_y + max_range)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('Acrobot Swing-up and Balance Control')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    
    # 绘制目标位置（虚线）
    target_line, = ax.plot([0, 0], [0, plant.l1 + plant.l2 + 0.2],
                           'g--', alpha=0.4, linewidth=2, label='Target')
    
    # 绘制元素
    link1_line, = ax.plot([], [], 'o-', lw=5, color='#1f77b4', markersize=10, label='Link 1')
    link2_line, = ax.plot([], [], 'o-', lw=5, color='#ff7f0e', markersize=10, label='Link 2')
    com1_marker, = ax.plot([], [], 's', color='#1f77b4', markersize=8, alpha=0.7)
    com2_marker, = ax.plot([], [], 's', color='#ff7f0e', markersize=8, alpha=0.7)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    mode_text = ax.text(0.02, 0.88, '', transform=ax.transAxes, fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    energy_text = ax.text(0.02, 0.81, '', transform=ax.transAxes, fontsize=9)
    
    ax.legend(loc='upper right', fontsize=9)
    
    l1, l2 = plant.l1, plant.l2
    lc1, lc2 = plant.lc1, plant.lc2
    
    def init():
        link1_line.set_data([], [])
        link2_line.set_data([], [])
        com1_marker.set_data([], [])
        com2_marker.set_data([], [])
        time_text.set_text('')
        mode_text.set_text('')
        energy_text.set_text('')
        return link1_line, link2_line, com1_marker, com2_marker, time_text, mode_text, energy_text
    
    def update(frame):
        q1, q2 = states[0, frame], states[1, frame]
        
        # 连杆端点位置
        x1 = l1 * np.sin(q1)
        y1 = -l1 * np.cos(q1)
        x2 = x1 + l2 * np.sin(q1 + q2)
        y2 = y1 - l2 * np.cos(q1 + q2)
        
        # 质心位置
        xc1 = lc1 * np.sin(q1)
        yc1 = -lc1 * np.cos(q1)
        xc2 = x1 + lc2 * np.sin(q1 + q2)
        yc2 = y1 - lc2 * np.cos(q1 + q2)
        
        link1_line.set_data([0, x1], [0, y1])
        link2_line.set_data([x1, x2], [y1, y2])
        com1_marker.set_data([xc1], [yc1])
        com2_marker.set_data([xc2], [yc2])
        
        time_text.set_text(f't = {t[frame]:.2f}s')
        
        return link1_line, link2_line, com1_marker, com2_marker, time_text, mode_text, energy_text
    
    # 降低帧率以减小 GIF 大小（同时保持视觉效果）
    skip = max(1, int(len(t) / (t[-1] * fps)))
    frames = range(0, len(t), skip)
    
    anim = FuncAnimation(fig, update, frames=frames,
                        init_func=init, blit=True,
                        interval=1000/fps)
    
    anim.save(filename, writer='pillow', fps=fps)
    print(f"动画已保存至 {filename}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return anim


def plot_snapshots(t, states, plant, filename='../figures/snapshots.png',
                   n_snapshots=6, show=False):
    """
    绘制仿真过程中的多个快照
    
    参数:
        t: 时间数组
        states: 状态轨迹
        plant: Acrobot 实例
        filename: 输出文件名
        n_snapshots: 快照数量
        show: 是否显示
    """
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    # 选择等间隔的快照帧
    indices = np.linspace(0, len(t) - 1, n_snapshots, dtype=int)
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        q1, q2 = states[0, idx], states[1, idx]
        
        # 计算位置
        x1 = plant.l1 * np.sin(q1)
        y1 = -plant.l1 * np.cos(q1)
        x2 = x1 + plant.l2 * np.sin(q1 + q2)
        y2 = y1 - plant.l2 * np.cos(q1 + q2)
        
        # 绘制
        ax.plot([0, x1], [0, y1], 'o-', lw=4, color='#1f77b4', markersize=8)
        ax.plot([x1, x2], [y1, y2], 'o-', lw=4, color='#ff7f0e', markersize=8)
        ax.plot([0, 0], [0, plant.l1 + plant.l2 + 0.2], 'g--', alpha=0.3, linewidth=2)
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.2, 2.2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f't = {t[idx]:.2f}s')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
    
    plt.suptitle('Acrobot Motion Snapshots', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"快照图已保存至 {filename}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_energy_breakdown(t, states, plant, filename='../figures/energy_breakdown.png',
                          show=False):
    """
    绘制能量分解图（动能 + 势能）
    """
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    ke = np.array([plant.kinetic_energy(states[:, i]) for i in range(states.shape[1])])
    pe = np.array([plant.potential_energy(states[:, i]) for i in range(states.shape[1])])
    total = ke + pe
    E_des = plant.desired_energy()
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # 能量组成
    ax = axes[0]
    ax.fill_between(t, 0, ke, alpha=0.5, label='Kinetic Energy')
    ax.fill_between(t, ke, ke + pe, alpha=0.5, label='Potential Energy')
    ax.plot(t, total, 'k-', linewidth=1.5, label='Total Energy')
    ax.axhline(E_des, color='r', linestyle='--', label=f'Target ({E_des:.2f} J)')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Energy [J]')
    ax.set_title('Energy Breakdown')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 能量误差
    ax = axes[1]
    energy_error = total - E_des
    ax.plot(t, energy_error, 'b-', linewidth=1.2)
    ax.axhline(0, color='r', linestyle='--', alpha=0.5)
    ax.fill_between(t, 0, energy_error, where=(energy_error >= 0),
                    alpha=0.3, color='green', label='Excess Energy')
    ax.fill_between(t, 0, energy_error, where=(energy_error < 0),
                    alpha=0.3, color='red', label='Energy Deficit')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Energy Error [J]')
    ax.set_title('Energy Error (E - E_target)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"能量分解图已保存至 {filename}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


if __name__ == "__main__":
    # 测试：先生成仿真数据，再创建动画
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    
    from simulate import simulate_acrobot_swing_up
    
    results = simulate_acrobot_swing_up(t_final=8.0, save_results=False)
    
    # 生成动画
    animate_acrobot(
        results['t'],
        results['x'],
        results['plant'],
        filename='../animations/acrobot_swing_up.gif',
        fps=30
    )
    
    # 生成快照
    plot_snapshots(
        results['t'],
        results['x'],
        results['plant'],
        filename='../figures/snapshots.png'
    )
    
    # 生成能量分解图
    plot_energy_breakdown(
        results['t'],
        results['x'],
        results['plant'],
        filename='../figures/energy_breakdown.png'
    )
