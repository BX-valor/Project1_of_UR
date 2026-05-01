"""
ablation.py - 消融实验

对比不同参数设置下控制器的性能：
1. R 矩阵对 LQR 的影响
2. 切换阈值的影响
3. Energy Shaping 参数 k_e 的影响
4. LQR Q 矩阵权重的影响
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from acrobot import Acrobot
from controllers import EnergyShapingController, LQRController, HybridController
from utils import linearize_system, evaluate_performance, compute_settling_time
from simulate import simulate_acrobot_swing_up

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 默认图片保存目录：acrobot_project/figures/
# 当前文件位于 acrobot_project/src/ablation.py
DEFAULT_FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')


def ablation_R_matrix(save_dir=None):
    """
    实验 1: R 矩阵对 LQR 的影响
    
    固定 Q，变化 R，观察 LQR 增益和平衡性能
    """
    if save_dir is None:
        save_dir = DEFAULT_FIGURES_DIR
    print("\n" + "=" * 60)
    print("消融实验 1: R 矩阵对 LQR 的影响")
    print("=" * 60)
    
    os.makedirs(save_dir, exist_ok=True)
    
    plant = Acrobot()
    x_up = np.array([np.pi, 0.0, 0.0, 0.0])
    A, B = linearize_system(plant, x_up, u_eq=0)
    
    R_values = [0.1, 1.0, 10.0, 100.0]
    Q = np.diag([10, 10, 1, 1])
    
    results = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, R_val in enumerate(R_values):
        R = np.array([[R_val]])
        
        lqr = LQRController(A, B, Q, R, x_up, u_max=10.0)
        
        # 从接近平衡点的初始状态测试 LQR
        x0 = np.array([np.pi - 0.3, 0.2, 0.0, 0.0])
        
        # 简单仿真（15秒，确保有足够的收敛时间）
        dt = 0.01
        t_eval = np.arange(0, 15.0, dt)
        states = [x0.copy()]
        controls = []
        x_current = x0.copy()
        
        for i in range(len(t_eval) - 1):
            u = lqr.compute_control(x_current)
            controls.append(u)
            
            from scipy.integrate import solve_ivp
            sol = solve_ivp(
                lambda t, x: plant.dynamics(x, u),
                [t_eval[i], t_eval[i+1]],
                x_current,
                method='RK45'
            )
            x_current = sol.y[:, -1]
            states.append(x_current.copy())
        
        controls.append(lqr.compute_control(x_current))
        
        t_arr = t_eval
        x_arr = np.array(states).T
        u_arr = np.array(controls)
        
        # 使用绝对阈值计算 settling_time（0.5 对应约 28° 总偏差）
        metrics = evaluate_performance(t_arr, x_arr, u_arr, x_up)
        metrics['settling_time'] = compute_settling_time(t_arr, x_arr, x_up, threshold=0.5, use_relative=False)
        metrics['R'] = R_val
        metrics['K'] = lqr.K.flatten()
        results.append(metrics)
        
        # 绘图
        ax = axes[idx]
        ax.plot(t_arr, x_arr[0, :], label=r'$\theta_1$', linewidth=1.2)
        ax.plot(t_arr, x_arr[1, :], label=r'$\theta_2$', linewidth=1.2)
        ax.axhline(np.pi, color='r', linestyle='--', alpha=0.3)
        ax.set_title(f'R = {R_val}\nK=[{lqr.K[0,0]:.2f}, {lqr.K[0,1]:.2f}, {lqr.K[0,2]:.2f}, {lqr.K[0,3]:.2f}]')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Angle [rad]')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Ablation: Effect of R Matrix on LQR Performance', fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(save_dir, 'ablation_R_matrix.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\nR 矩阵消融图已保存至: {save_path}")
    plt.show()
    
    # 打印结果表格
    print("\n结果汇总:")
    print(f"{'R':>8} {'Settling':>10} {'Overshoot':>10} {'Max Torque':>12} {'Steady Error':>13}")
    print("-" * 60)
    for r in results:
        st = f"{r['settling_time']:.2f}" if r['settling_time'] is not None else "N/A"
        print(f"{r['R']:>8.1f} {st:>10} {r['overshoot']:>9.2f}% {r['max_torque']:>11.2f} {r['steady_error']:>12.4f}")
    
    return results


def ablation_switching_threshold(save_dir=None):
    """
    实验 2: 切换阈值的影响
    
    测试不同切换阈值对整体 swing-up and balance 的影响
    """
    if save_dir is None:
        save_dir = DEFAULT_FIGURES_DIR
    print("\n" + "=" * 60)
    print("消融实验 2: 切换阈值的影响")
    print("=" * 60)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 优化阈值选择：实际切换时 cost-to-go ≈ 27，在 5~30 范围内取点
    # 可观察到"不切换→延迟切换→正常切换"的完整过渡
    thresholds = [5, 10, 15, 20, 30, 50]
    all_results = []
    
    # 2x4 布局：6 个阈值实验 + 1 个汇总对比 + 1 个空位
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flatten()
    # 隐藏多余的空子图（放在最右下角）
    axes[-1].set_visible(False)
    
    for idx, thresh in enumerate(thresholds):
        print(f"\n测试阈值: {thresh}")
        result = simulate_acrobot_swing_up(
            t_final=10.0,
            dt=0.005,
            k_e=30.0,
            k_p=50.0,
            k_d=10.0,
            switching_threshold=thresh,
            u_max=10.0,
            save_results=False
        )
        
        t = result['t']
        x = result['x']
        u = result['u']
        modes = result['modes']
        
        # 统计切换次数
        switch_count = len(result['switch_history'])
        
        metrics = result['metrics'].copy()
        # Swing-up 实验放宽 settling time 阈值到 1.0（稳态误差通常在 0.3~0.6）
        metrics['settling_time'] = compute_settling_time(t, x, np.array([np.pi, 0, 0, 0]), threshold=1.0, use_relative=False)
        metrics['threshold'] = thresh
        metrics['switch_count'] = switch_count
        all_results.append(metrics)
        
        # 异常检测：未切换提示
        if switch_count == 0:
            print(f"  [警告] 阈值={thresh} 时未发生切换，LQR 未接管控制。可能阈值过小或系统未进入吸引域。")
        
        # 绘图
        ax = axes[idx]
        ax.plot(t, x[0, :], label=r'$\theta_1$', linewidth=1.2)
        ax.plot(t, x[1, :], label=r'$\theta_2$', linewidth=1.2)
        ax.axhline(np.pi, color='r', linestyle='--', alpha=0.3)
        
        # 标记切换点
        for t_switch, direction, cost in result['switch_history']:
            ax.axvline(t_switch, color='purple', linestyle=':', alpha=0.5)
        
        ax.set_title(f'Threshold = {thresh}\nSwitches: {switch_count}, Settling: {metrics["settling_time"]:.2f}s' if metrics['settling_time'] else f'Threshold = {thresh}\nSwitches: {switch_count}, Settling: N/A')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Angle [rad]')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 留一个子图放汇总对比（放在第二行倒数第二个，与其他子图排列整齐）
    ax = axes[6]
    thresholds_list = [r['threshold'] for r in all_results]
    settling_times = [r['settling_time'] if r['settling_time'] is not None else 10.0 for r in all_results]
    max_torques = [r['max_torque'] for r in all_results]
    
    ax2 = ax.twinx()
    ax.bar(np.arange(len(thresholds_list)) - 0.2, settling_times, 0.4, label='Settling Time', color='steelblue')
    ax2.bar(np.arange(len(thresholds_list)) + 0.2, max_torques, 0.4, label='Max Torque', color='coral')
    ax.set_xticks(range(len(thresholds_list)))
    ax.set_xticklabels([str(t) for t in thresholds_list])
    ax.set_xlabel('Switching Threshold')
    ax.set_ylabel('Settling Time [s]', color='steelblue')
    ax2.set_ylabel('Max Torque [Nm]', color='coral')
    ax.set_title('Performance vs Threshold')
    ax.grid(True, alpha=0.3)
    
    # 合并图例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.suptitle('Ablation: Effect of Switching Threshold', fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(save_dir, 'ablation_switching_threshold.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\n切换阈值消融图已保存至: {save_path}")
    plt.show()
    
    # 打印结果表格
    print("\n结果汇总:")
    print(f"{'Threshold':>10} {'Settling':>10} {'Switches':>10} {'Max Torque':>12} {'Steady Error':>13}")
    print("-" * 60)
    for r in all_results:
        st = f"{r['settling_time']:.2f}" if r['settling_time'] is not None else "N/A"
        print(f"{r['threshold']:>10} {st:>10} {r['switch_count']:>10} {r['max_torque']:>11.2f} {r['steady_error']:>12.4f}")
    
    return all_results


def ablation_k_e(save_dir=None):
    """
    实验 3: Energy Shaping 参数 k_e 的影响
    
    单独测试 swing-up 阶段（不启用 LQR），观察能量收敛速度
    """
    if save_dir is None:
        save_dir = DEFAULT_FIGURES_DIR
    print("\n" + "=" * 60)
    print("消融实验 3: 能量增益 k_e 的影响")
    print("=" * 60)
    
    os.makedirs(save_dir, exist_ok=True)
    
    plant = Acrobot()
    x0 = np.array([0.0, 0.0, 0.05, 0.0])
    E_des = plant.desired_energy()
    
    k_e_values = [10, 30, 50, 80, 120]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    all_energy_profiles = []
    
    for idx, k_e in enumerate(k_e_values):
        print(f"\n测试 k_e: {k_e}")
        
        ctrl = EnergyShapingController(plant, k_e=k_e, k_p=5.0, k_d=1.0, u_max=10.0)
        
        # 仿真（仅 swing-up，不切换）
        dt = 0.01
        t_eval = np.arange(0, 8.0, dt)
        states = [x0.copy()]
        controls = []
        energies = []
        x_current = x0.copy()
        
        for i in range(len(t_eval) - 1):
            u = ctrl.compute_control(x_current)
            controls.append(u)
            energies.append(plant.energy(x_current))
            
            from scipy.integrate import solve_ivp
            sol = solve_ivp(
                lambda t, x: plant.dynamics(x, u),
                [t_eval[i], t_eval[i+1]],
                x_current,
                method='RK45'
            )
            x_current = sol.y[:, -1]
            states.append(x_current.copy())
        
        energies.append(plant.energy(x_current))
        
        t_arr = t_eval
        x_arr = np.array(states).T
        u_arr = np.array(controls)
        e_arr = np.array(energies)
        
        all_energy_profiles.append((k_e, t_arr, e_arr, x_arr, u_arr))
        
        # 绘图 - 能量
        ax = axes[idx]
        ax.plot(t_arr, e_arr, label='Energy', linewidth=1.5)
        ax.axhline(E_des, color='r', linestyle='--', label=f'Target ({E_des:.2f} J)')
        ax.set_title(f'k_e = {k_e}')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Energy [J]')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 留一个子图放能量对比
    ax = axes[-1]
    for k_e, t_arr, e_arr, _, _ in all_energy_profiles:
        ax.plot(t_arr, e_arr, label=f'k_e={k_e}', linewidth=1.2)
    ax.axhline(E_des, color='r', linestyle='--', alpha=0.5, label='Target')
    ax.set_title('Energy Convergence Comparison')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Energy [J]')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Ablation: Effect of Energy Gain k_e', fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(save_dir, 'ablation_k_e.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\nk_e 消融图已保存至: {save_path}")
    plt.show()
    
    # 打印能量收敛分析（使用平均力矩代替最大力矩，因为最大力矩常达到饱和）
    print("\n能量收敛分析:")
    print(f"{'k_e':>8} {'90% Time':>10} {'Max Energy':>12} {'Max Torque':>12} {'Status':>12}")
    print("-" * 65)
    for k_e, t_arr, e_arr, x_arr, u_arr in all_energy_profiles:
        threshold = 0.9 * E_des
        above_threshold = np.where(e_arr >= threshold)[0]
        if len(above_threshold) > 0:
            t_90 = t_arr[above_threshold[0]]
        else:
            t_90 = "N/A"
        max_e = np.max(e_arr)
        max_u = np.max(np.abs(u_arr))
        t_str = f"{t_90:.2f}" if isinstance(t_90, float) else t_90
        
        # 异常检测
        status = "OK"
        if max_e > 2 * E_des:
            status = "超调警告"
        elif t_90 == "N/A":
            status = "未收敛"
        
        print(f"{k_e:>8} {t_str:>10} {max_e:>11.2f} {max_u:>11.2f} {status:>12}")
    
    return all_energy_profiles


def ablation_Q_weights(save_dir=None):
    """
    实验 4: LQR Q 矩阵对角元素的影响
    
    测试不同角度权重对 LQR 性能的影响
    """
    if save_dir is None:
        save_dir = DEFAULT_FIGURES_DIR
    print("\n" + "=" * 60)
    print("消融实验 4: LQR Q 矩阵权重的影响")
    print("=" * 60)
    
    os.makedirs(save_dir, exist_ok=True)
    
    plant = Acrobot()
    x_up = np.array([np.pi, 0.0, 0.0, 0.0])
    A, B = linearize_system(plant, x_up, u_eq=0)
    R = np.array([[1.0]])
    
    # 不同的 Q 对角权重
    Q_configs = [
        ('Angle Heavy', np.diag([50, 50, 1, 1])),
        ('Velocity Heavy', np.diag([1, 1, 50, 50])),
        ('Balanced', np.diag([10, 10, 10, 10])),
        ('Shoulder Focus', np.diag([50, 1, 10, 1])),
        ('Elbow Focus', np.diag([1, 50, 1, 10])),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    results = []
    
    for idx, (name, Q) in enumerate(Q_configs):
        print(f"\n测试 Q 配置: {name}")
        
        lqr = LQRController(A, B, Q, R, x_up, u_max=10.0)
        
        # 测试 LQR
        x0 = np.array([np.pi - 0.3, 0.2, 0.0, 0.0])
        dt = 0.01
        t_eval = np.arange(0, 15.0, dt)
        states = [x0.copy()]
        controls = []
        x_current = x0.copy()
        
        for i in range(len(t_eval) - 1):
            u = lqr.compute_control(x_current)
            controls.append(u)
            
            from scipy.integrate import solve_ivp
            sol = solve_ivp(
                lambda t, x: plant.dynamics(x, u),
                [t_eval[i], t_eval[i+1]],
                x_current,
                method='RK45'
            )
            x_current = sol.y[:, -1]
            states.append(x_current.copy())
        
        controls.append(lqr.compute_control(x_current))
        
        t_arr = t_eval
        x_arr = np.array(states).T
        u_arr = np.array(controls)
        
        metrics = evaluate_performance(t_arr, x_arr, u_arr, x_up)
        # 使用绝对阈值重新计算 settling_time
        metrics['settling_time'] = compute_settling_time(t_arr, x_arr, x_up, threshold=0.5, use_relative=False)
        metrics['name'] = name
        metrics['Q_diag'] = np.diag(Q)
        results.append(metrics)
        
        # 绘图
        ax = axes[idx]
        ax.plot(t_arr, x_arr[0, :], label=r'$\theta_1$', linewidth=1.2)
        ax.plot(t_arr, x_arr[1, :], label=r'$\theta_2$', linewidth=1.2)
        ax.axhline(np.pi, color='r', linestyle='--', alpha=0.3)
        ax.set_title(f'{name}\nQ=diag({np.diag(Q)})')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Angle [rad]')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 汇总对比
    ax = axes[-1]
    names = [r['name'] for r in results]
    settling = [r['settling_time'] if r['settling_time'] is not None else 5.0 for r in results]
    steady_errors = [r['steady_error'] for r in results]
    
    ax2 = ax.twinx()
    bars1 = ax.bar(np.arange(len(names)) - 0.2, settling, 0.4, label='Settling Time', color='steelblue')
    bars2 = ax2.bar(np.arange(len(names)) + 0.2, steady_errors, 0.4, label='Steady Error', color='coral')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylabel('Settling Time [s]', color='steelblue')
    ax2.set_ylabel('Steady Error', color='coral')
    ax.set_title('Performance Comparison')
    ax.grid(True, alpha=0.3)
    
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.suptitle('Ablation: Effect of LQR Q Matrix Weights', fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(save_dir, 'ablation_Q_weights.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\nQ 权重消融图已保存至: {save_path}")
    plt.show()
    
    # 打印结果
    print("\n结果汇总:")
    print(f"{'Config':>16} {'Settling':>10} {'Overshoot':>10} {'Max Torque':>12} {'Steady Error':>13}")
    print("-" * 65)
    for r in results:
        st = f"{r['settling_time']:.2f}" if r['settling_time'] is not None else "N/A"
        print(f"{r['name']:>16} {st:>10} {r['overshoot']:>9.2f}% {r['max_torque']:>11.2f} {r['steady_error']:>12.4f}")
    
    return results


def run_all_ablations():
    """
    运行所有消融实验
    """
    print("\n" + "=" * 70)
    print("开始运行所有消融实验")
    print("=" * 70)
    
    results_r = ablation_R_matrix()
    results_thresh = ablation_switching_threshold()
    results_ke = ablation_k_e()
    results_q = ablation_Q_weights()
    
    print("\n" + "=" * 70)
    print("所有消融实验完成！")
    print("=" * 70)
    
    return {
        'R_matrix': results_r,
        'switching_threshold': results_thresh,
        'k_e': results_ke,
        'Q_weights': results_q
    }


if __name__ == "__main__":
    run_all_ablations()
