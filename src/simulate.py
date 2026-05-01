"""
simulate.py - 主仿真脚本

实现 Acrobot Swing-up and Balance 的完整闭环仿真。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os
import sys

from acrobot import Acrobot
from controllers import EnergyShapingController, LQRController, HybridController
from utils import linearize_system, evaluate_performance

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 默认图片保存目录：acrobot_project/figures/
# 当前文件位于 acrobot_project/src/simulate.py
DEFAULT_FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')


def simulate_acrobot_swing_up(plant=None, x0=None, t_final=10.0, dt=0.01,
                               k_e=30.0, k_p=50.0, k_d=10.0,
                               Q=None, R=None,
                               switching_threshold=20.0,
                               state_constraints=None,
                               u_max=10.0,
                               save_results=True):
    """
    运行 Acrobot Swing-up and Balance 仿真
    
    参数:
        plant: Acrobot 实例，None 则使用默认参数
        x0: 初始状态，None 则使用向下悬挂状态
        t_final: 仿真总时间 [s]
        dt: 时间步长 [s]
        k_e, k_p, k_d: Energy Shaping 参数
        Q, R: LQR 权重矩阵
        switching_threshold: 切换阈值
        u_max: 力矩限幅
        save_results: 是否保存结果图
    
    返回:
        results: dict 包含仿真结果
    """
    # 默认系统参数
    if plant is None:
        plant = Acrobot(
            m1=1.0, m2=1.0,
            l1=1.0, l2=1.0,
            lc1=0.5, lc2=0.5,
            g=9.81
        )
    
    # 初始状态和目标状态
    if x0 is None:
        x0 = np.array([0.0, 0.0, 0.05, 0.0])  # 向下悬挂，带微小初始速度以启动 swing-up
    x_up = np.array([np.pi, 0.0, 0.0, 0.0])  # 竖直向上
    
    print("=" * 60)
    print("Acrobot Swing-up and Balance 仿真")
    print("=" * 60)
    print(f"初始状态: {x0}")
    print(f"目标状态: {x_up}")
    print(f"初始能量: {plant.energy(x0):.4f} J")
    print(f"目标能量: {plant.desired_energy():.4f} J")
    
    # LQR 设计
    print("\n--- LQR 设计 ---")
    A, B = linearize_system(plant, x_up, u_eq=0)
    
    if Q is None:
        Q = np.diag([10, 10, 1, 1])
    if R is None:
        R = np.array([[1.0]])
    
    lqr = LQRController(A, B, Q, R, x_up, u_max=u_max)
    
    # Energy Shaping 控制器
    print("\n--- Energy Shaping 控制器 ---")
    energy_ctrl = EnergyShapingController(
        plant,
        k_e=k_e,
        k_p=k_p,
        k_d=k_d,
        u_max=u_max
    )
    
    # 混合控制器
    print("\n--- 混合控制器 ---")
    hybrid = HybridController(
        energy_ctrl,
        lqr,
        switching_threshold=switching_threshold,
        hysteresis_factor=2.0,
        state_constraints=state_constraints
    )
    
    # 仿真设置
    t_span = (0, t_final)
    t_eval = np.arange(t_span[0], t_span[1], dt)
    
    # 历史记录
    states_history = [x0.copy()]
    controls_history = []
    controller_modes = []
    energies_history = []
    times_history = [0.0]
    
    # 使用逐步积分以便记录中间结果
    x_current = x0.copy()
    
    for i in range(len(t_eval) - 1):
        t_start = t_eval[i]
        t_end = t_eval[i + 1]
        
        # 计算控制输入
        u, mode = hybrid.compute_control(x_current, t=t_start)
        
        # 记录
        controls_history.append(u)
        controller_modes.append(mode)
        energies_history.append(plant.energy(x_current))
        
        # 积分一步
        sol = solve_ivp(
            lambda t, x: plant.dynamics(x, u),
            [t_start, t_end],
            x_current,
            method='RK45',
            max_step=dt
        )
        
        x_current = sol.y[:, -1]
        states_history.append(x_current.copy())
        times_history.append(t_end)
    
    # 最终状态记录
    u, mode = hybrid.compute_control(x_current, t=t_eval[-1])
    controls_history.append(u)
    controller_modes.append(mode)
    energies_history.append(plant.energy(x_current))
    
    # 转换为 numpy 数组
    t_arr = np.array(times_history)
    x_arr = np.array(states_history).T  # (4, N)
    u_arr = np.array(controls_history)
    e_arr = np.array(energies_history)
    
    # 评估性能
    print("\n--- 性能评估 ---")
    metrics = evaluate_performance(t_arr, x_arr, u_arr, x_up)
    for key, val in metrics.items():
        if val is not None:
            print(f"{key}: {val:.4f}")
        else:
            print(f"{key}: N/A")
    
    # 切换历史
    switch_history = hybrid.get_switch_history()
    if switch_history:
        print(f"\n切换次数: {len(switch_history)}")
        for t_switch, direction, cost in switch_history:
            print(f"  t={t_switch:.3f}s: {direction} (cost={cost:.4f})")
    
    results = {
        't': t_arr,
        'x': x_arr,
        'u': u_arr,
        'energies': e_arr,
        'modes': controller_modes,
        'plant': plant,
        'x_eq': x_up,
        'metrics': metrics,
        'switch_history': switch_history,
        'lqr': lqr,
        'hybrid': hybrid
    }
    
    if save_results:
        plot_simulation_results(results)
    
    return results


def plot_simulation_results(results, save_dir=None, show=True):
    if save_dir is None:
        save_dir = DEFAULT_FIGURES_DIR
    """
    绘制仿真结果
    
    参数:
        results: simulate_acrobot_swing_up 返回的字典
        save_dir: 保存目录
        show: 是否显示图形
    """
    t = results['t']
    x = results['x']
    u = results['u']
    energies = results['energies']
    modes = results['modes']
    plant = results['plant']
    x_eq = results['x_eq']
    
    E_des = plant.desired_energy()
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    # 状态轨迹 - 角度
    ax = axes[0, 0]
    ax.plot(t, x[0, :], label=r'$\theta_1$ (Shoulder)', linewidth=1.5)
    ax.plot(t, x[1, :], label=r'$\theta_2$ (Elbow)', linewidth=1.5)
    ax.axhline(np.pi, color='r', linestyle='--', alpha=0.5, label='Target')
    ax.axhline(0, color='r', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Angle [rad]')
    ax.legend()
    ax.set_title('Joint Angles')
    ax.grid(True, alpha=0.3)
    
    # 状态轨迹 - 角速度
    ax = axes[0, 1]
    ax.plot(t, x[2, :], label=r'$\dot{\theta}_1$', linewidth=1.5)
    ax.plot(t, x[3, :], label=r'$\dot{\theta}_2$', linewidth=1.5)
    ax.axhline(0, color='r', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Angular Velocity [rad/s]')
    ax.legend()
    ax.set_title('Angular Velocities')
    ax.grid(True, alpha=0.3)
    
    # 控制输入
    ax = axes[1, 0]
    ax.plot(t[:len(u)], u, 'g', linewidth=1.2)
    ax.axhline(plant.I1, color='r', linestyle='--', alpha=0.3, label='u_max')
    ax.axhline(-plant.I1, color='r', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Torque [Nm]')
    ax.set_title('Control Input')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 能量
    ax = axes[1, 1]
    ax.plot(t[:len(energies)], energies, label='Current Energy', linewidth=1.5)
    ax.axhline(E_des, color='r', linestyle='--', label=f'Target Energy ({E_des:.2f} J)')
    ax.axhline(plant.energy(x_eq), color='g', linestyle=':', alpha=0.7, label='Upward Equilibrium')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Energy [J]')
    ax.legend()
    ax.set_title('System Energy')
    ax.grid(True, alpha=0.3)
    
    # 控制器模式
    ax = axes[2, 0]
    mode_numeric = [1 if m == "LQR" else 0 for m in modes]
    ax.fill_between(t[:len(modes)], 0, mode_numeric, alpha=0.3, color='purple')
    ax.plot(t[:len(modes)], mode_numeric, 'purple', linewidth=1.0, label='Controller Mode')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('0=ES, 1=LQR')
    ax.set_title('Controller Switching')
    ax.set_ylim(-0.1, 1.2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 相图 (theta1 vs theta1_dot)
    ax = axes[2, 1]
    ax.plot(x[0, :], x[2, :], alpha=0.6, linewidth=1.0)
    ax.scatter([np.pi], [0], color='r', s=150, marker='*', label='Target', zorder=5)
    ax.scatter([x[0, 0]], [x[2, 0]], color='g', s=100, marker='o', label='Start', zorder=5)
    ax.set_xlabel(r'$\theta_1$ [rad]')
    ax.set_ylabel(r'$\dot{\theta}_1$ [rad/s]')
    ax.set_title('Phase Portrait ($\\theta_1$)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'swing_up_results.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\n结果图已保存至: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def run_multiple_simulations():
    """
    运行多个不同初始条件的仿真，测试鲁棒性
    """
    print("\n" + "=" * 60)
    print("多初始条件鲁棒性测试")
    print("=" * 60)
    
    plant = Acrobot()
    x_up = np.array([np.pi, 0.0, 0.0, 0.0])
    
    # 不同初始条件
    initial_conditions = [
        np.array([0.0, 0.0, 0.0, 0.0]),           # 标准向下
        np.array([0.2, 0.0, 0.0, 0.0]),           # 轻微偏移
        np.array([0.0, 0.5, 0.0, 0.0]),           # 第二关节偏移
        np.array([0.0, 0.0, 0.5, 0.0]),           # 有初始速度
        np.array([np.pi, 0.3, 0.0, 0.0]),         # 接近平衡点但第二关节偏
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, x0 in enumerate(initial_conditions):
        print(f"\n初始条件 {idx+1}: {x0}")
        result = simulate_acrobot_swing_up(
            plant=plant, x0=x0, t_final=8.0,
            save_results=False
        )
        
        t = result['t']
        x = result['x']
        
        ax = axes[idx]
        ax.plot(t, x[0, :], label=r'$\theta_1$', linewidth=1.2)
        ax.plot(t, x[1, :], label=r'$\theta_2$', linewidth=1.2)
        ax.axhline(np.pi, color='r', linestyle='--', alpha=0.4)
        ax.set_title(f'IC {idx+1}: [{x0[0]:.1f}, {x0[1]:.1f}, {x0[2]:.1f}, {x0[3]:.1f}]')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Angle [rad]')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(DEFAULT_FIGURES_DIR, 'robustness_test.png')
    os.makedirs(DEFAULT_FIGURES_DIR, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\n鲁棒性测试图已保存至: {save_path}")
    plt.show()


if __name__ == "__main__":
    # 主仿真
    results = simulate_acrobot_swing_up(
        t_final=10.0,
        dt=0.005,
        k_e=50.0,
        k_p=5.0,
        k_d=1.0,
        switching_threshold=50.0,
        u_max=10.0,
        save_results=True
    )
    
    # 可选：运行多初始条件测试
    # run_multiple_simulations()
